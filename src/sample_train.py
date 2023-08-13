import pickle

import javalang
import torch

from models.tbcnn import TBCNNRegressor


def parse_java_files():
    """
    Parses multiple Java files into ASTs.
    :return: List of ASTs corresponding to the Java files.
    """

    with open("../data/rdf4j_tree.pickle", "rb") as fp:
        data = pickle.load(fp)

    return data.values()


def get_node_type(node, vocab_size):
    """
    Map AST node type to an integer. Use Python's built-in `hash` function and modulus
    operation to get an integer in [0, vocab_size - 1].

    :return: Integer representation of the node type.
    """
    return hash(node.__class__.__name__) % vocab_size


def ast_to_graph(node, nodes, edges, parent=None, vocab_size=20):
    """
    Converts a Java AST node to a graph representation.
    """

    current_id = len(nodes)
    nodes[current_id] = {'type': torch.tensor(get_node_type(node, vocab_size=vocab_size))}
    if parent is not None:
        edges.append((parent, current_id))

    if hasattr(node, 'children'):
        for child in node.children:
            if isinstance(child, javalang.ast.Node):
                ast_to_graph(child, nodes, edges, current_id)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, javalang.ast.Node):
                        ast_to_graph(item, nodes, edges, current_id)


def trees_to_graphs(trees, vocab_size=20):
    """
    Converts a list of ASTs to graph representations.

    :return: Tuple of lists representing nodes and edges of the graphs.
    """
    all_nodes = []
    all_edges = []
    for tree in trees:
        nodes = {}
        edges = []
        ast_to_graph(tree, nodes, edges, vocab_size=vocab_size)
        all_nodes.append(nodes)
        all_edges.append(edges)
    return all_nodes, all_edges


if __name__ == '__main__':
    vocab_size = 20
    trees = parse_java_files()
    all_nodes, all_edges = trees_to_graphs(trees, vocab_size=vocab_size)

    model = TBCNNRegressor(x_size=100, h_size=50, dropout=0.5, vocab_size=vocab_size)

    # Processing each graph:
    for nodes, edges in zip(all_nodes, all_edges):
        output = model(nodes, edges)
        print(output)
        # Add loss computation, backpropagation, and optimizer steps here
