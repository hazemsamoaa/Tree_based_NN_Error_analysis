import sys
import argparse
import pprint

from utils import read_csv, read_pickle, write_pickle


sys.setrecursionlimit(6000)


def main(args):
    """
    Main function to read files and prepare the data.

    Args:
        args (argparse.Namespace): The namespace containing command-line arguments.
    """
    y = read_csv(args.csv_file_path)
    x = read_pickle(args.pickle_file_path)

    data = []
    for j_file in x:
        value = y[y["Test case"] == j_file]["Runtime in ms"].tolist()[0]
        data.append({
            'tree': x[j_file],
            'metadata': {'value': value}
        })

    write_pickle(data, args.output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read CSV and Pickle files and prepare data.')
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--pickle_file_path', type=str, required=True, help='Path to the input Pickle file.')
    parser.add_argument('--output_file_path', type=str, required=True, help='Path to the output Pickle file.')

    args = parser.parse_args()
    main(args)

    java_algo = read_pickle(args.output_file_path)
    pprint.pp(java_algo[0])
    print(dir(java_algo[0]["tree"]))
    print()
    for i, algo in enumerate(java_algo):
        if i > 1:
            break

        print(algo["tree"])

    print()
