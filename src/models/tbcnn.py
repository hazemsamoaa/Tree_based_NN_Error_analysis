"""
The code inspired based on
https://github.com/crestonbunch/tbcnn/blob/master/classifier/tbcnn/network.py
"""

import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim




class ConvLayer(nn.Module):
    def __init__(self, feature_size, output_size=100):
        super().__init__()

        self.feature_size = feature_size
        self.conv_wt = nn.Parameter(torch.randn(feature_size, output_size) * math.sqrt(1. / feature_size))
        self.conv_wr = nn.Parameter(torch.randn(feature_size, output_size) * math.sqrt(1. / feature_size))
        self.conv_wl = nn.Parameter(torch.randn(feature_size, output_size) * math.sqrt(1. / feature_size))
        self.conv_b = nn.Parameter(torch.randn(output_size) * math.sqrt(2. / feature_size))

    def children_tensor(self, nodes, children, feature_size):
        """
        Build the children tensor from the input nodes and child lookup.
        """
        batch_size, num_nodes, _ = nodes.size()
        max_children = children.size(2)

        # Replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        zero_vecs = torch.zeros((batch_size, 1, feature_size), device=nodes.device)
        vector_lookup = torch.cat([zero_vecs, nodes[:, 1:, :]], dim=1)

        # Prepend the batch indices to the 4th dimension of children
        batch_indices = torch.arange(0, batch_size).view(batch_size, 1, 1, 1)
        batch_indices = batch_indices.repeat(1, num_nodes, max_children, 1)

        # Convert to long tensor for index operations
        batch_indices = batch_indices.long()
        children = children.unsqueeze(3).long()

        # Concatenate batch indices and children indices
        children = torch.cat([batch_indices, children], dim=3)

        # Gather vectors based on children indices
        # Create a tensor of shape (batch_size, num_nodes, max_children, 2),
        # where the last dimension contains both batch and node indices
        children_indices = children.view(-1, 2)

        # Use these indices to get vectors from vector_lookup
        children_vectors = vector_lookup[children_indices[:, 0], children_indices[:, 1]]

        # Reshape to (batch_size, num_nodes, max_children, feature_size)
        children_vectors = children_vectors.view(batch_size, num_nodes, max_children, feature_size)

        return children_vectors

    def eta_t(self, children):
        """
        Compute weight matrix for how much each vector belongs to the 'top'
        """
        batch_size, max_tree_size, max_children = children.size()

        # Create a tensor with ones in the first column and zeros in other columns
        ones = torch.ones((max_tree_size, 1), device=children.device)
        zeros = torch.zeros((max_tree_size, max_children), device=children.device)
        combined = torch.cat([ones, zeros], dim=1)

        # Expand dimensions and tile to create the final eta_t tensor
        eta_t = combined.unsqueeze(0).repeat(batch_size, 1, 1)

        return eta_t

    def eta_r(self, children, t_coef):
        """
        Compute weight matrix for how much each vector belongs to the 'right'
        """
        batch_size, max_tree_size, max_children = children.size()
        children = children.float()

        # Count non-zero elements in the 'children' tensor along the last axis
        num_siblings = (children != 0).sum(dim=2, keepdim=True).float()

        # Create tensor filled with num_siblings repeated along the last axis
        num_siblings = num_siblings.repeat(1, 1, max_children + 1)

        # Create a mask of 1's and 0's where 1 means there is a child there
        mask = torch.cat([torch.zeros((batch_size, max_tree_size, 1)), (children != 0).float()], dim=2)

        # Create child indices for every tree
        child_indices = torch.arange(-1.0, max_children, 1.0).float()
        child_indices = child_indices.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_tree_size, 1)
        child_indices *= mask

        # Create tensor for special case when num_siblings == 1
        singles = torch.cat([
            torch.zeros((batch_size, max_tree_size, 1)),
            torch.full((batch_size, max_tree_size, 1), 0.5),
            torch.zeros((batch_size, max_tree_size, max_children - 1))
        ], dim=2)

        # Compute the eta_r tensor
        eta_r = torch.where(
            num_siblings == 1.0,
            singles,
            (1.0 - t_coef) * (child_indices / (num_siblings - 1.0))
        )

        return eta_r

    def eta_l(self, children, coef_t, coef_r):
        """
        Compute weight matrix for how much each vector belongs to the 'left'
        """
        batch_size, max_tree_size, max_children = children.size()
        children = children.float()

        # Create a mask of 1's and 0's where 1 means there is a child there
        mask = torch.cat([torch.zeros((batch_size, max_tree_size, 1)), (children != 0).float()], dim=2)

        # Compute the eta_l tensor
        eta_l = (1.0 - coef_t) * (1.0 - coef_r) * mask

        return eta_l

    def conv_step(self, nodes, children, feature_size, conv_wt, conv_wr, conv_wl, conv_b):
        # children_vectors will have shape
        # (batch_size x max_tree_size x max_children x feature_size)
        children_vectors = self.children_tensor(nodes, children, feature_size)
        nodes = torch.unsqueeze(nodes, dim=2)
        tree_tensor = torch.cat([nodes, children_vectors], dim=2)

        c_t = self.eta_t(children)
        c_r = self.eta_r(children, c_t)
        c_l = self.eta_l(children, c_t, c_r)

        coef = torch.stack([c_t, c_r, c_l], dim=3)
        weights = torch.stack([conv_wt, conv_wr, conv_wl], dim=0)

        batch_size, max_tree_size, max_children = children.size()
        feature_size = tree_tensor.size(-1)
        output_size = weights.size(-1)

        # Reshape for matrix multiplication
        x = batch_size * max_tree_size
        y = max_children + 1
        result = tree_tensor.view(x, y, feature_size)
        coef = coef.view(x, y, 3)

        # MatMul with transpose
        result = torch.matmul(result.permute(0, 2, 1), coef)

        # Reshape the result back
        result = result.view(batch_size, max_tree_size, 3, feature_size)

        # Apply weights using tensor dot operation
        result = torch.einsum('ijkl,klm->ijm', result, weights)

        # Apply bias and tanh activation
        result = torch.tanh(result + conv_b)
        return result

    def forward(self, nodes, children):
        r = self.conv_step(nodes, children, self.feature_size, self.conv_wt, self.conv_wr, self.conv_wl, self.conv_b)
        return r


class TreeConvNet(nn.Module):
    def __init__(self, feature_size, label_size, num_conv=1, output_size=100):
        super().__init__()

        self.feature_size = feature_size
        self.label_size = label_size

        # Convolution layers
        self.nodes_list = nn.ModuleList([
            ConvLayer(feature_size, output_size)
            for _ in range(num_conv)
        ])

        # Hidden layer
        self.hidden = nn.Linear(output_size, label_size)

    def pooling_layer(self, nodes):
        return torch.max(nodes, dim=1).values

    def forward(self, nodes, children):
        # nodes: batch_size x max_tree_size x feature_size
        # children: batch_size x max_tree_size x max_children
        x = nodes
        for layer in self.nodes_list:
            x = layer(x, children)

        pooled = self.pooling_layer(x)
        # hidden = F.relu(self.hidden(pooled))
        hidden = self.hidden(pooled)
        return hidden


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




LEARN_RATE = 0.001
EPOCHS = 50
CHECKPOINT_EVERY = 100
BATCH_SIZE = 1


def training(args):
    with open(args.infile, 'rb') as fh:
        trees, test_trees, labels, label_scaler = pickle.load(fh)

    with open(args.embedfile, 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh)
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
        y_pred_list = torch.tensor(y_pred_list)
        y_true_list = torch.tensor(y_true_list)

        p_corr = pearson_corr(y_pred_list, y_true_list)
        print(f'\nEpoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / total_samples:.4f}, P-CORR: {p_corr}\n')

    return net


if __name__ == '__main__':
    args = AttrDict({
        "infile": "../../data/rdf4j/java_algorithm_trees.pkl",
        "embedfile": "../../data/rdf4j/java_algorithm_vectors.pkl",
    })

    training(args)
