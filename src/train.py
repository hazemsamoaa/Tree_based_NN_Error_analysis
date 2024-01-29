import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from data_utils import find_java_files, parse_java_code_to_ast
from models.code2vec.config import Config as Code2vecConfig
from models.code2vec.net import Code2vecNet
from models.code2vec.prepare_data import prepare_data as code2vec_prepare_data
from models.code2vec.trainer import trainer as code2vec_trainer
from models.tree_cnn.embedding import TreeCNNEmbedding
from models.tree_cnn.prepare_data import prepare_nodes as tree_cnn_prepare_nodes
from models.tree_cnn.prepare_data import prepare_trees as tree_cnn_prepare_trees
from models.tree_cnn.tbcnn import TreeConvNet
from models.tree_cnn.trainer import node_trainer as tree_cnn_node_trainer
from models.tree_cnn.trainer import trainer as tree_cnn_trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

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
        if model == "code2vec":
            # _output_dir = os.path.join(args.output_dir, f"tree_cnn_{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}")
            _output_dir = os.path.join(args.output_dir, f"tree_cnn")
            os.makedirs(_output_dir, exist_ok=True)

            logger.info(f"starting on extractign representation based on {model} ...")
            code2vec_config = Code2vecConfig(set_defaults=True, load_from_args=True, verify=True, args=args)
            train_data, test_data = code2vec_prepare_data(train, code2vec_config, test=test)

            model = Code2vecNet(feature_size=train_data[0]["representation"].shape[1]).to(device)
            code2vec_trainer(
                model,
                train=train_data,
                test=test_data,
                y_scaler=y_scaler,
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

    # Code2Vec
    parser.add_argument("-d", "--data", dest="data_path", help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path", help="path to test file", metavar="FILE", required=False, default='')
    parser.add_argument("-s", "--save", dest="save_path", help="path to save the model file", metavar="FILE", required=False)
    parser.add_argument("-w2v", "--save_word2v", dest="save_w2v", help="path to save the tokens embeddings file", metavar="FILE", required=False)
    parser.add_argument("-t2v", "--save_target2v", dest="save_t2v", help="path to save the targets embeddings file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path", help="path to load the model from", metavar="FILE", required=False)
    parser.add_argument('--save_w2v', dest='save_w2v', required=False, help="save word (token) vectors in word2vec format")
    parser.add_argument('--save_t2v', dest='save_t2v', required=False, help="save target vectors in word2vec format")
    parser.add_argument('--export_code_vectors', action='store_true', required=False, help="export code vectors for the given examples")
    parser.add_argument('--release', action='store_true', help='if specified and loading a trained model, release the loaded model for a lower model size.')
    parser.add_argument('--predict', action='store_true', help='execute the interactive prediction shell')
    parser.add_argument("-fw", "--framework", dest="dl_framework", choices=['keras', 'tensorflow'], default='tensorflow', help="deep learning framework to use.")
    parser.add_argument("-v", "--verbose", dest="verbose_mode", type=int, required=False, default=1, help="verbose mode (should be in {0,1,2}).")
    parser.add_argument("-lp", "--logs-path", dest="logs_path", metavar="FILE", required=False, help="path to store logs into. if not given logs are not saved to file.")
    parser.add_argument('-tb', '--tensorboard', dest='use_tensorboard', action='store_true', help='use tensorboard during training')
    parser.add_argument('--in_dir', type=str, required=False, help='Java files directory.')
    parser.add_argument('--out_dir', type=str, required=False, help='Code vector directory.')
    parser.add_argument('--jar_path', type=str, default="./scripts/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar")
    parser.add_argument('--max_contexts', type=int, default=200)
    parser.add_argument('--max_path_length', type=int, default=8)
    parser.add_argument('--max_path_width', type=int, default=2)

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
