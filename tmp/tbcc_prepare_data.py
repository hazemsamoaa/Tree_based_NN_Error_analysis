import argparse
import os
import sys
import numpy as np
import pandas as pd
from pycparser import c_parser
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from models.tbcc.tree import trans_to_sequences
from utils import read_csv, write_pickle
from sklearn.model_selection import train_test_split
import re
import javalang
sys.setrecursionlimit(6000)


def remove_comments(text):
    """Remove C-style /*comments*/ from a string."""
    p = r'/\*[^*]*\*+([^/*][^*]*\*+)*/|("(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\'|.[^/"\'\\]*)'
    return ''.join(m.group(2) for m in re.finditer(p, text, re.M | re.S) if m.group(2))

def main(args):
    """
    Main function to read files and prepare the data.

    Args:
        args (argparse.Namespace): The namespace containing command-line arguments.
    """

    parser = c_parser.CParser()

    vocabulary = dict()
    inverse_vocabulary = ['<unk>']

    data = read_csv(args.csv_file_path)
    error_rows = []
    dataset = []

    for index, row in tqdm(data.iterrows(), total=len(data)):

        q2n = []
        code_path = os.path.join(os.path.dirname(args.csv_file_path), row["Test case"])
        output = row["Runtime in ms"]

        if not os.path.exists(code_path):
            continue

        code = ""
        with open(code_path, "r", encoding="utf-8") as fp:
            code = remove_comments(fp.read())

        try:
            tokens = list(javalang.tokenizer.tokenize(code))
            ast = javalang.parse.parse(code)
            ast = trans_to_sequences(ast)
        except Exception as e:
            print(e)
            error_rows.append(index)
            continue

        tokens_for_sents = ['CLS'] + ast + ['SEP']
        if len(tokens_for_sents) > args.max_seq_length:
            error_rows.append(index)
            continue

        for word in tokens_for_sents:

            # Check for unwanted words
            # if word in stops and word not in word2vec.wv.vocab:
            #     continue

            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                q2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(word)
            else:
                q2n.append(vocabulary[word])

        dataset.append({
            "id": index,
            "code": code,
            "repr": q2n,
            "output": output
        })

    print(f"error_rows: {error_rows}")
    dataset = pd.DataFrame(dataset)
    dataset.to_json(os.path.join(args.output_dir, "data.json"), indent=2, orient="records")

    max_seq_length = dataset.repr.map(lambda x: len(x)).max()
    print('max_seq_length: %d' % max_seq_length)
    print('vocabulary: %d' % (len(vocabulary) + 1))
    write_pickle(vocabulary, os.path.join(args.output_dir, "vocabulary.pkl"))

    train, test = train_test_split(dataset, test_size=0.2, random_state=101)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_json(os.path.join(args.output_dir, "train.json"), indent=2, orient="records")
    test.to_json(os.path.join(args.output_dir, "test.json"), indent=2, orient="records")

    label_scaler = MinMaxScaler()
    label_scaler.fit(np.expand_dims(train["output"].values, axis=-1))
    write_pickle(label_scaler, os.path.join(args.output_dir, "label_scaler.pkl"))

    with open(os.path.join(args.output_dir, "params.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"--vocab_size {len(vocabulary) + 1} --max_seq_length {max_seq_length}")


if __name__ == "__main__":
    #  python src/tbcc_prepare_data.py --csv_file_path ./data/tbcc/code_classification_data_for_Ccode.csv --output_dir ./data/tbcc/
    parser = argparse.ArgumentParser(description='Read CSV and Pickle files and prepare data.')
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output Pickle file.')

    args = parser.parse_args()

    main(args)
