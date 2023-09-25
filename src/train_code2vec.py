import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# from metrics import pearson_corr, pearson_corr_v2
from metrics import pearson_corr_v2 as pearson_corr
from models.code2vec.net import Code2vecNet
from utils import read_pickle


def batch_samples(data, batch_size):
    """Batch samples from a generator"""
    code_vectors, labels = [], []
    samples = 0
    for x in data:
        code_vectors.append(x["code_vector"])
        labels.append(x["metadata"]["value"])
        samples += 1
        if samples >= batch_size:
            yield _pad_batch(code_vectors, labels)
            code_vectors, labels = [], []
            samples = 0


def _pad_batch(code_vectors, labels):
    if not code_vectors:
        return [], []

    max_rows = max(arr.shape[0] for arr in code_vectors)
    max_cols = code_vectors[0].shape[1]
    padded_code_vectors = [
        np.expand_dims(np.vstack([arr, np.zeros((max_rows - arr.shape[0], max_cols))]), axis=0)
        for arr in code_vectors
    ]

    labels = np.array(labels)

    return np.concatenate(padded_code_vectors, axis=0), labels


LEARN_RATE = 1e-4
EPOCHS = 1
CHECKPOINT_EVERY = 100
BATCH_SIZE = 1


def training(args):
    data = read_pickle(args.infile)
    num_feats = data[0]["code_vector"].shape[1]

    train, test = train_test_split(data, test_size=0.15, random_state=101)
    y_train = np.array([r["metadata"]["value"] for r in train]).reshape(-1, 1)

    label_scaler = MinMaxScaler()
    label_scaler.fit(y_train)

    print(f"number of data: {len(data)} {data[0].keys()}")
    print(f"number of train: {len(train)}")
    print(f"number of test: {len(test)}")
    print(f"number of features: {num_feats}")

    # Setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the network
    print(f"num_feats: {num_feats}")
    net = Code2vecNet(feature_size=num_feats).to(device)
    print(f"NET: {net}")
    # Cross Entropy loss
    criterion = nn.MSELoss()

    # Optimizer
    # optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE)
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)

    # Initialize step count
    total_samples = len(train)
    step_count = 0

    # Training loop
    net.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        # P-CORR
        y_pred_list = []
        y_true_list = []

        for i, batch in enumerate(batch_samples(train, batch_size=BATCH_SIZE)):
            batch_code_vectors, batch_labels = batch

            batch_code_vectors = torch.from_numpy(batch_code_vectors).float().to(device)
            batch_labels = label_scaler.transform(batch_labels.reshape(-1, 1))
            batch_labels = torch.tensor(batch_labels).float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = net(batch_code_vectors)
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

    net.eval()
    with torch.no_grad():
        total_samples = len(test)
        total_loss = 0.0

        # P-CORR
        y_pred_list = []
        y_true_list = []

        for i, batch in enumerate(batch_samples(test, batch_size=BATCH_SIZE)):
            batch_code_vectors, batch_labels = batch

            batch_code_vectors = torch.from_numpy(batch_code_vectors).float().to(device)
            batch_labels = label_scaler.transform(batch_labels.reshape(-1, 1))
            batch_labels = torch.tensor(batch_labels).float().to(device)

            # Forward pass
            logits = net(batch_code_vectors)
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
    # python src/train_code2vec.py --infile ./data/java/code2vec.pkl --net_outfile ./checkpoints/code2vec/
    parser = argparse.ArgumentParser(description='Train a neural network on tree-structured data.')
    parser.add_argument('--infile', type=str, required=True, help='Input file with training data.')
    parser.add_argument('--net_outfile', type=str, required=True, help='Output file for the neural network model.')

    args = parser.parse_args()
    if args.net_outfile:
        os.makedirs(args.net_outfile, exist_ok=True)
    training(args)
