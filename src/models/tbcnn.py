"""
The code inspired based on
https://github.com/jacobwwh/tbcnn-dgl/blob/main/tbcnn.py
"""

import torch
import torch.nn as nn
from torch.nn.functional import relu


class TBCNNCell(torch.nn.Module):
    """
    Tree-based Convolutional Neural Network (TBCNN) cell.
    This is the building block of the TBCNN, handling message passing and reduction within the network.
    """

    def __init__(self, x_size, h_size):
        """
        Initialize the TBCNN cell.

        Parameters:
        x_size: int, size of the input.
        h_size: int, size of the hidden state.
        """
        super(TBCNNCell, self).__init__()
        self.W_left = nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_right = nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_top = nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.b_conv = nn.Parameter(torch.zeros(1, h_size), requires_grad=True)

        self.W_left.data.uniform_(-0.1, 0.1)
        self.W_right.data.uniform_(-0.1, 0.1)
        self.W_top.data.uniform_(-0.1, 0.1)
        self.b_conv.data.uniform_(-0.1, 0.1)

    def message_passing(self, edge_list, h):
        """
        Perform the message passing step, propagating information along the edges of the graph.

        Parameters:
        edge_list: list of tuples, each tuple representing an edge from source to target node.
        h: dict, node hidden states.
        """
        message_dict = {}
        for edge in edge_list:
            src, dest = edge
            if dest not in message_dict:
                message_dict[dest] = []
            message_dict[dest].append(h[src])
        return message_dict

    def reduce_func(self, nodes, h):
        """
        Perform the reduction step, aggregating the messages at their target nodes and updating the nodes' states.

        Parameters:
        nodes: dict, node states.
        h: dict, node hidden states.
        """
        new_h = {}
        for node in nodes:
            if node in h:
                child_h = torch.stack(h[node], dim=0)
                child_nums = child_h.size()[0]
                if child_nums == 1:
                    c_s = torch.matmul(child_h, self.W_left)
                    children_state = c_s.squeeze(1)
                    new_h[node] = relu(children_state + torch.matmul(nodes[node]['h'], self.W_top) + self.b_conv)
                else:
                    left_weight = torch.tensor([(child_nums - 1 - i) / (child_nums - 1) for i in range(child_nums)]).to(self.W_left.device)
                    right_weight = torch.tensor([i / (child_nums - 1) for i in range(child_nums)]).to(self.W_left.device)

                    child_h_left = torch.matmul(child_h, self.W_left)
                    child_h_left *= left_weight.unsqueeze(1)
                    child_h_right = torch.matmul(child_h, self.W_right)
                    child_h_right *= right_weight.unsqueeze(1)

                    children_state = child_h_left + child_h_right
                    children_state = children_state.sum(dim=0)
                    new_h[node] = relu(children_state + torch.matmul(nodes[node]['h'], self.W_top) + self.b_conv)
        return new_h


class TBCNNRegressor(torch.nn.Module):
    """
    Tree-based Convolutional Neural Network (TBCNN) regressor.
    This is the overall network, which applies the TBCNN cell in a recursive manner to the input graph, and includes the final regression layer.
    """

    def __init__(self, x_size, h_size, dropout, vocab_size, num_layers=1, n_outputs=1):
        """
        Initialize the TBCNN.

        Parameters:
        x_size: int, size of the input.
        h_size: int, size of the hidden state.
        dropout: float, dropout rate for regularization.
        n_classes: int, number of classes for classification.
        vocab_size: int, size of the vocabulary (number of unique node types).
        num_layers: int, number of layers in the network.
        """
        super(TBCNNRegressor, self).__init__()
        self.x_size = x_size
        self.dropout = torch.nn.Dropout(dropout)
        self.layers = nn.ModuleList(TBCNNCell(x_size, h_size) for _ in range(num_layers))
        self.embeddings = nn.Embedding(vocab_size, x_size)
        self.output = nn.Linear(h_size, n_outputs)

    def forward(self, nodes, edge_list):
        """
        Forward pass through the network.

        Parameters:
        nodes: dict, node states.
        edge_list: list of tuples, each tuple representing an edge from source to target node.
        root_ids: list of root node ids, used if multiple graphs are being processed at once.
        """
        h = {node_id: self.embeddings(features['type']) for node_id, features in nodes.items()}
        nodes = {node_id: {'h': h[node_id]} for node_id in nodes.keys()}

        for i in range(len(self.layers)):
            h = self.layers[i].message_passing(edge_list, h)
            h = self.layers[i].reduce_func(nodes, h)

        output = torch.max(torch.stack([h_value for h_value in h.values()]), dim=0)[0]
        output = self.output(output)
        return output


# if __name__ == '__main__':
#     import torch
#     import random
#     import numpy as np
#
#     SEED = 0
#     torch.manual_seed(SEED)
#     random.seed(SEED)
#     np.random.seed(SEED)
#
#     nodes = {
#         0: {'type': torch.tensor(1)},
#         1: {'type': torch.tensor(2)},
#         2: {'type': torch.tensor(3)},
#         3: {'type': torch.tensor(4)},
#         4: {'type': torch.tensor(5)},
#     }
#     edge_list = [(0, 1), (1, 2), (2, 3), (3, 4)]
#
#     # Define the model
#     model = TBCNNRegressor(x_size=100, h_size=50, dropout=0.5, vocab_size=20)
#
#     # Forward pass
#     output = model(nodes, edge_list)
#     print(output)
