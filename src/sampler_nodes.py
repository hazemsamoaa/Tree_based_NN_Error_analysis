"""Parse nodes from a given data source."""

import argparse
import pprint
from collections import defaultdict

import javalang
import javalang.ast

from utils import AttrDict, read_pickle, write_pickle


def parse(args):
    """Parse the tree data from a pickle file and create samples.

    Args:
        args (AttrDict): The arguments containing paths and limits for processing.
    """
    print('Loading pickle file')
    data_source = read_pickle(args.infile)
    print('Pickle load finished')

    node_counts = defaultdict(int)
    samples = []

    has_capacity = lambda x: args.per_node < 0 or node_counts[x] < args.per_node
    can_add_more = lambda: args.limit < 0 or len(samples) < args.limit

    for item in data_source:
        root = item['tree']
        new_samples = [
            {
                'node': _name(root),
                'parent': None,
                'children': [_name(x) for x in _iter_child_nodes(root)]
            }
        ]
        gen_samples = lambda x: new_samples.extend(_create_samples(x))
        _traverse_tree(root, gen_samples)
        for sample in new_samples:
            if has_capacity(sample['node']):
                samples.append(sample)
                node_counts[sample['node']] += 1
            if not can_add_more():
                break
        if not can_add_more():
            break

    print('dumping sample')
    write_pickle(samples, args.outfile)
    print('Sampled node counts:')
    print(node_counts)
    print('Total: %d' % sum(node_counts.values()))


def _create_samples(node):
    """Create samples based on the node's children.

    Args:
        node (javalang.ast.Node): The node for which to create samples.

    Returns:
        list: A list of samples created.
    """

    samples = []
    for child in _iter_child_nodes(node):
        sample = {
            "node": _name(child),
            "parent": _name(node),
            "children": [_name(x) for x in _iter_child_nodes(child)]
        }
        samples.append(sample)

    return samples


def _traverse_tree(tree, callback):
    """Traverse the tree using breadth-first search and apply a callback.

    Args:
        tree (javalang.ast.Node): The root node of the tree.
        callback (function): The callback function to apply.
    """

    queue = [tree]
    while queue:
        current_node = queue.pop(0)
        children = list(_iter_child_nodes(current_node))
        queue.extend(children)
        callback(current_node)


def _iter_child_nodes(node):
    """Generate child nodes for a given node.

    Args:
        node (javalang.ast.Node): The node for which to generate child nodes.

    Yields:
        javalang.ast.Node: A child node.
    """

    if not isinstance(node, javalang.ast.Node):
        return

    for attr, value in node.__dict__.items():
        if isinstance(value, javalang.ast.Node):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, javalang.ast.Node):
                    yield item


def _name(node):
    """Retrieve the name of a node's class.

    Args:
        node (javalang.ast.Node): The node for which to retrieve the class name.

    Returns:
        str: The name of the node's class.
    """

    return type(node).__name__


if __name__ == '__main__':
    # python src/sampler_nodes.py --infile "./data/rdf4j/java_algorithms.pkl" --outfile "./data/rdf4j/java_algorithm_nodes.pkl" --limit -1 --per_node -1
    parser = argparse.ArgumentParser(description='Parse nodes from a given data source.')
    parser.add_argument('--infile', type=str, required=True, help='Input pickle file path.')
    parser.add_argument('--outfile', type=str, required=True, help='Output pickle file path.')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples. Set to -1 for no limit.')
    parser.add_argument('--per_node', type=int, default=-1, help='Limit samples per node. Set to -1 for no limit.')

    args = parser.parse_args()
    parse(args)

    java_nodes = read_pickle(args.outfile)
    pprint.pp(java_nodes[0])
    NODE_MAPS = []
    for java_node in java_nodes:
        node = java_node["node"]
        parent = java_node["node"]

        if node not in NODE_MAPS:
            NODE_MAPS.append(node)

        if parent not in NODE_MAPS:
            NODE_MAPS.append(parent)
