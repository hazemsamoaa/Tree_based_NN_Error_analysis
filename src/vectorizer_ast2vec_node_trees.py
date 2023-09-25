"""
A script for training a neural network on tree-structured data.
"""

import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from utils import read_pickle

# Constants
NUM_FEATURES = 100
BATCH_SIZE = 256
EPOCHS = 10
LEARN_RATE = 0.01
HIDDEN_NODES = 100
CHECKPOINT_EVERY = 100

# List of nodes
NODE_LIST = [
    'CompilationUnit', 'PackageDeclaration', 'Import', 'ClassDeclaration', 'FieldDeclaration',
    'MethodDeclaration', 'ReferenceType', 'VariableDeclarator', 'Annotation', 'StatementExpression',
    'TryStatement', 'MethodInvocation', 'Assignment', 'LocalVariableDeclaration', 'CatchClause',
    'MemberReference', 'ClassCreator', 'CatchClauseParameter', 'Literal', 'FormalParameter',
    'TypeArgument', 'ClassReference', 'LambdaExpression', 'AssertStatement', 'ReturnStatement',
    'BinaryOperation', 'MethodReference', 'Cast', 'ElementValuePair', 'BasicType', 'IfStatement',
    'BlockStatement', 'ThrowStatement', 'ElementArrayValue', 'ForStatement', 'EnhancedForControl',
    'VariableDeclaration', 'ConstructorDeclaration', 'SuperConstructorInvocation', 'ArrayCreator',
    'ArraySelector', 'ArrayInitializer', 'SuperMethodInvocation', 'TryResource', 'WhileStatement',
    'This', 'Statement', 'ForControl', 'BreakStatement', 'InferredFormalParameter', 'TypeParameter',
    'TernaryExpression', 'ContinueStatement'
]

NODE_MAP = {x: i for (i, x) in enumerate(NODE_LIST)}


class Net(nn.Module):
    """A simple feedforward neural network."""

    def __init__(self, num_feats=NUM_FEATURES, hidden_size=HIDDEN_NODES, num_classes=len(NODE_MAP)):
        super(Net, self).__init__()

        # Embedding layer
        self.embeddings = nn.Embedding(num_classes, num_feats)

        # Hidden layer
        self.hidden = nn.Linear(num_feats, hidden_size)

        # Softmax layer
        self.softmax = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embeddings(x)

        # Hidden layer with tanh activation
        x = torch.tanh(self.hidden(x))

        # Logits
        logits = self.softmax(x)

        return logits


def batch_samples(samples, batch_size):
    """
    Yields batches of samples.

    Args:
        samples: List of samples.
        batch_size: Size of each batch.

    Yields:
        A batch of samples.
    """

    batch = ([], [])
    count = 0
    index_of = lambda x: NODE_MAP[x]
    for sample in samples:
        if sample['parent'] is not None:
            batch[0].append(index_of(sample['node']))
            batch[1].append(index_of(sample['parent']))
            count += 1
            if count >= batch_size:
                yield batch
                batch, count = ([], []), 0


def training(args):
    """
    Train a neural network.

    Args:
        args: A dictionary of command-line arguments.

    Returns:
        A trained neural network model.
    """

    samples = read_pickle(args.infile)

    # Setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create the network
    net = Net()
    net.to(device)

    # Cross Entropy loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE)

    # Initialize step count
    step_count = 0

    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0.0
        sample_gen = batch_samples(samples, BATCH_SIZE)
        total_samples = len(samples)

        for i, batch in enumerate(sample_gen):
            inputs, labels = batch
            inputs = torch.LongTensor(inputs).to(device)
            labels = torch.LongTensor(labels).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = net(inputs)

            # Compute loss
            loss = criterion(logits, labels)
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
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / total_samples:.4f}')

    return net


if __name__ == '__main__':
    # python src/vectorizer_ast2vec_node_trees.py --infile ./data/rdf4j/java_algorithm_nodes.pkl --vectors_outfile ./data/rdf4j/java_algorithm_vectors.pkl --net_outfile ./data/rdf4j/java_algorithm_net.pth
    parser = argparse.ArgumentParser(description='Train a neural network on tree-structured data.')
    parser.add_argument('--infile', type=str, required=True, help='Input file with training data.')
    parser.add_argument('--vectors_outfile', type=str, required=True, help='Output file for learned vectors.')
    parser.add_argument('--net_outfile', type=str, required=True, help='Output file for the neural network model.')

    args = parser.parse_args()
    model = training(args)

    embedding_matrix = model.embeddings.weight.data.cpu().numpy()
    with open(args.vectors_outfile, "wb") as fp:
        pickle.dump((embedding_matrix, NODE_MAP), fp)

    torch.save(model.state_dict(), args.net_outfile)
