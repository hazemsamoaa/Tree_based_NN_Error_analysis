import pickle
import random
import sys
from collections import defaultdict
import argparse
import javalang
from sklearn.preprocessing import MinMaxScaler
from utils import read_pickle, write_pickle



def _iter_child_nodes(node):
    """
    Yields the child nodes of a given node.

    Args:
        node: The parent node.

    Yields:
        The child nodes of the parent node.
    """
    for attr, value in node.__dict__.items():
        if isinstance(value, javalang.ast.Node):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, javalang.ast.Node):
                    yield item


def parse(args):
    """
    Parses, filters, and serializes tree structures.

    Args:
        args: A dictionary of command-line arguments.
    """
    print('Loading pickle file')
    sys.setrecursionlimit(1000000)
    data_source = read_pickle(args.infile)
    print('Pickle file load finished')

    train_samples = []
    test_samples = []

    train_counts = defaultdict(int)
    test_counts = defaultdict(int)

    label_list = []

    print(f"len of data_source: {len(data_source)}")
    # raise

    for item in data_source:
        root = item['tree']
        label = item['metadata'][args.label_key]
        sample, size = _traverse_tree(root)

        if size > args.maxsize or size < args.minsize:
            continue

        roll = random.randint(0, 100)

        datum = {'tree': sample, 'label': label}

        if roll < args.test:
            test_samples.append(datum)
            test_counts[label] += 1
        else:
            label_list.append([label])
            train_samples.append(datum)
            train_counts[label] += 1
    
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    labels = [0.0]
    label_scaler = MinMaxScaler()
    label_scaler.fit(label_list)
    print('Dumping sample')
    write_pickle((train_samples, test_samples, labels, label_scaler), args.outfile)
    print('dump finished')
    print('Sampled tree counts: ')
    print('Training:', len(train_counts))
    print('Testing:', len(test_counts))


def _traverse_tree(root):
    """
    Traverse a tree to produce a JSON-like structure.

    Args:
        root: The root node of the tree.

    Returns:
        A tuple containing the JSON-like structure and the number of nodes.
    """
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),
        "children": []
    }
    queue_json = [root_json]

    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        current_node_json = queue_json.pop(0)

        children = list(_iter_child_nodes(current_node))
        queue.extend(children)

        for child in children:
            child_json = {
                "node": _name(child),
                "children": []
            }
            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes


def _name(node):
    """
    Get the name of a node.

    Args:
        node: The node.

    Returns:
        The name of the node.
    """
    return type(node).__name__


if __name__ == '__main__':
    # python sampler_trees.py --infile ./data/rdf4j/java_algorithms.pkl --outfile ./data/rdf4j/java_algorithm_trees.pkl --label_key "value" --minsize 100 --maxsize 2000 --test 15
    parser = argparse.ArgumentParser(description='Parse, filter, and serialize trees.')
    parser.add_argument('--infile', type=str, required=True, help='Input pickle file path.')
    parser.add_argument('--outfile', type=str, required=True, help='Output pickle file path.')
    parser.add_argument('--label_key', type=str, default='value', help='Key to use for labels in metadata.')
    parser.add_argument('--minsize', type=int, default=100, help='Minimum size for tree.')
    parser.add_argument('--maxsize', type=int, default=2000, help='Maximum size for tree.')
    parser.add_argument('--test', type=int, default=15, help='Percentage of data to be used for testing.')

    args = parser.parse_args()
    parse(args)

    # java_algo = read_pickle(args.outfile)
    # print(java_algo)
