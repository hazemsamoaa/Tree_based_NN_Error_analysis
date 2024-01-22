import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from data_utils import find_java_files, parse_java_code_to_ast
from models.tree_cnn.embedding import TreeCNNEmbedding
from models.tree_cnn.trainer import (
    node_trainer as tree_cnn_node_trainer,
    trainer as tree_cnn_trainer
)
from models.tree_cnn.tbcnn import TreeConvNet
from models.tree_cnn.prepare_data import (
    prepare_nodes as tree_cnn_prepare_nodes,
    prepare_trees as tree_cnn_prepare_trees
)
from utils import AttrDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cfg = AttrDict({
    "n_samples_per_log": 3,
    "models": ["tree_cnn", "code2vec", "transformer_tree", "tree_gen"],
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    java_files = find_java_files(args.data_dir)[:100]
    data_info = pd.read_csv(os.path.join(args.data_dir, "data_info.csv"))
    data = []

    for java_file in tqdm(java_files):
        file_name, full_path, file_type = java_file
        row = data_info[data_info["Test case"] == file_name].to_dict("records")

        if len(row) > 0:
            row = row[0]

            with open(os.path.join(full_path), "r", encoding="utf-8") as f:
                java_code = f.read()
                tree = parse_java_code_to_ast(java_code, logger)

            if tree:
                data.append({
                    "name": file_name,
                    "path": full_path,
                    "type": file_type,
                    "tree": tree,
                    "y": row["Runtime in ms"]
                })

    logger.info(f"Found {len(data)} for this experiment out of {len(java_files)}!")
    train, test = train_test_split(data, test_size=args.test_size, random_state=args.seed, stratify=list(map(lambda x: x["type"], data)))
    y_scaler = MinMaxScaler()
    y_scaler.fit(list(map(lambda x: [x["y"]], train)))

    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")

    for model in args.train_on:
        if model not in cfg.models:
            logger.error(f"unknown model for {model}")

        logger.info(f"training on {model} ...")
        if model == "tree_cnn":
            # _output_dir = os.path.join(args.output_dir, f"tree_cnn_{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}")
            _output_dir = os.path.join(args.output_dir, f"tree_cnn")
            os.makedirs(_output_dir, exist_ok=True)

            logger.info(f"starting on {model} ...")

            nodes, node_samples = tree_cnn_prepare_nodes(data=list(map(lambda x: x["tree"], data)), per_node=-1, limit=-1)
            logger.info(f"We have {len(nodes)} nodes: {nodes.keys()}")

            tree_cnn_embedding = TreeCNNEmbedding(num_classes=len(nodes.keys()), num_feats=100, hidden_size=100).to(device)
            node_map = {node: i for i, node in enumerate(nodes)}
            tree_cnn_embedding = tree_cnn_node_trainer(
                node_samples, tree_cnn_embedding,
                node_map=node_map,
                device=device, lr=args.lr, batch_size=args.batch_size, epochs=args.repr_epochs, checkpoint=args.checkpoint, output_dir=_output_dir)
            embeddings = tree_cnn_embedding.embeddings.weight.data.cpu().numpy()

            train_trees = tree_cnn_prepare_trees(data=list(map(lambda x: x["tree"], train)), minsize=-1, maxsize=-1)
            train_trees = [train_trees, list(map(lambda x: x["y"], train))]
            test_trees = tree_cnn_prepare_trees(data=list(map(lambda x: x["tree"], test)), minsize=-1, maxsize=-1)
            test_trees = [test_trees, list(map(lambda x: x["y"], test))]
            logger.info(f"Train size: {len(train_trees[0])}, Test size: {len(test_trees[0])}")

            model = TreeConvNet(feature_size=len(embeddings[0]), label_size=1, num_conv=1, output_size=100).to(device)
            tree_cnn_trainer(
                model,
                train_trees=train_trees,
                test_trees=test_trees,
                y_scaler=y_scaler,
                embeddings=embeddings,
                embed_lookup=node_map,
                device=device,
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, checkpoint=args.checkpoint, output_dir=_output_dir
            )
        else:
            logger.info(f"failed on {model} ...")


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--data_dir', type=str, help='provide dataset directory', required=True)
    parser.add_argument('--test_size', type=float, default=0.2, help="proportion of dataset to training and testing sets")
    parser.add_argument('--train_on', nargs='+', help='train on these models', required=True)

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--repr_epochs', type=int, default=10, help='number of epochs to train representation (default: 10)')
    parser.add_argument('--output_dir', type=str, default="", help='provide output directory')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--checkpoint', type=int, default=-1, help='number of checkpoint to log (default: -1)')

    # parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    # setting the seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # preprocess
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
