"""
The code inspired based on
https://github.com/crestonbunch/tbcnn/blob/master/classifier/tbcnn/network.py
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from metrics import pearson_corr
from metrics import pearson_corr_v2 as pearson_corr
from models.tbcnn.tbcnn import TreeConvNet
from utils import read_pickle


def gen_samples(trees, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    for tree in trees:
        nodes = []
        children = []
        label = [tree['label']]

        queue = [(tree['tree'], -1)]
        while queue:
            node, parent_ind = queue.pop(0)
            node_ind = len(nodes)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            nodes.append(vectors[vector_lookup[node['node']]])

        yield nodes, children, label


def batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels = [], [], []
    samples = 0
    for n, c, l in gen:
        nodes.append(n)
        children.append(c)
        labels.append(l)
        samples += 1
        if samples >= batch_size:
            yield _pad_batch(nodes, children, labels)
            nodes, children, labels = [], [], []
            samples = 0

    if nodes:
        yield _pad_batch(nodes, children, labels)


def _pad_batch(nodes, children, labels):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children, labels


def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]


LEARN_RATE = 1e-4
EPOCHS = 1000
CHECKPOINT_EVERY = 100
BATCH_SIZE = 1


def training(args):
    trees, test_trees, labels, label_scaler = read_pickle(args.infile)
    embeddings, embed_lookup = read_pickle(args.embedfile)
    num_feats = len(embeddings[0])

    print(f"number of trees: {len(trees)}")
    print(f"number of test trees: {len(test_trees)}")
    print(f"number of labels: {len(labels)}")
    print(f"label scaler: {label_scaler}")
    print(f"number of features: {num_feats}")

    # Setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the network
    print(f"num_feats: {num_feats}")
    net = TreeConvNet(num_feats, len(labels)).to(device)
    print(f"NET: {net}")

    # Cross Entropy loss
    criterion = nn.MSELoss()

    # Optimizer
    # optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE)
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)

    # Initialize step count
    total_samples = len(trees)
    step_count = 0

    # Training loop
    net.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        # P-CORR
        y_pred_list = []
        y_true_list = []

        for i, batch in enumerate(batch_samples(
            gen_samples(trees, labels, embeddings, embed_lookup),
            BATCH_SIZE
        )):
            nodes, children, batch_labels = batch
            nodes = torch.from_numpy(np.array(nodes)).to(device)
            children = torch.LongTensor(children).to(device)
            batch_labels = label_scaler.transform(batch_labels)
            batch_labels = torch.tensor(batch_labels).float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = net(nodes, children)

            y_pred_list.append(logits.item())
            y_true_list.append(batch_labels.item())

            # Compute loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update step count
            step_count += 1

            # Checkpoint based on steps
            if step_count % CHECKPOINT_EVERY == 0:
                print(f'Step [{step_count}], Loss: {total_loss / (i + 1):.4f}')
                # torch.save(net.state_dict(), f'checkpoint_step_{step_count}.pth')

        # Log per-epoch statistics
        y_pred_list = torch.tensor(label_scaler.inverse_transform(np.array(y_pred_list).reshape(1, -1))).squeeze()
        y_true_list = torch.tensor(label_scaler.inverse_transform(np.array(y_true_list).reshape(1, -1))).squeeze()

        p_corr = pearson_corr(y_pred_list, y_true_list)
        train_msg = f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / total_samples:.4f}, P-CORR: {p_corr}\n'
        print(train_msg)

        if args.net_outfile:
            with open(os.path.join(args.net_outfile, 'train_pred.txt'), "w", encoding="utf-8") as f:
                f.write(f"TRUE\tPRED\n")
                for i, j in zip(y_true_list.tolist(), y_pred_list.tolist()):
                    f.write(f"{i}\t{j}\n")

    # do eval
    total_samples = len(test_trees)
    total_loss = 0.0

    # P-CORR
    y_pred_list = []
    y_true_list = []

    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(batch_samples(
            gen_samples(test_trees, labels, embeddings, embed_lookup),
            BATCH_SIZE
        )):
            nodes, children, batch_labels = batch
            nodes = torch.from_numpy(np.array(nodes)).to(device)
            children = torch.LongTensor(children).to(device)
            batch_labels = label_scaler.transform(batch_labels)
            batch_labels = torch.tensor(batch_labels).float().to(device)

            # Forward pass
            logits = net(nodes, children)

            y_pred_list.append(logits.item())
            y_true_list.append(batch_labels.item())

            # Compute loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

        # Log per-epoch statistics
        y_pred_list = torch.tensor(label_scaler.inverse_transform(np.array(y_pred_list).reshape(1, -1))).squeeze()
        y_true_list = torch.tensor(label_scaler.inverse_transform(np.array(y_true_list).reshape(1, -1))).squeeze()

        p_corr = pearson_corr(y_pred_list, y_true_list)
        eval_msg = f'Loss: {total_loss / total_samples:.4f}, P-CORR: {p_corr}\n'
        print(eval_msg)

    if args.net_outfile:
        torch.save(net.state_dict(), os.path.join(args.net_outfile, f'model.pth'))
        with open(os.path.join(args.net_outfile, 'output.txt'), "w", encoding="utf-8") as f:
            f.write(f"TRAIN: {train_msg}")
            f.write(f" EVAL: {eval_msg}")

        with open(os.path.join(args.net_outfile, 'eval_pred.txt'), "w", encoding="utf-8") as f:
            f.write(f"TRUE\tPRED\n")
            for i, j in zip(y_true_list.tolist(), y_pred_list.tolist()):
                f.write(f"{i}\t{j}\n")

    return net


if __name__ == '__main__':
    #  python src/train_tbcnn.py --infile ./data/rdf4j/java_algorithm_trees.pkl --embedfile ./data/rdf4j/java_algorithm_vectors.pkl --net_outfile ./checkpoints/tbcnn/
    parser = argparse.ArgumentParser(description='Train a neural network on tree-structured data.')
    parser.add_argument('--infile', type=str, required=True, help='Input file with training data.')
    parser.add_argument('--embedfile', type=str, required=True, help='Embedding file for learned vectors.')
    parser.add_argument('--net_outfile', type=str, required=True, help='Output file for the neural network model.')

    args = parser.parse_args()
    if args.net_outfile:
        os.makedirs(args.net_outfile, exist_ok=True)
    net = training(args)
