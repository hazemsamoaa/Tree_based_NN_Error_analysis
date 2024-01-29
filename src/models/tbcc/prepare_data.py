import argparse
import os
import re
import sys

import javalang
import numpy as np
import pandas as pd
from models.tbcc.tree import trans_to_sequences
from pycparser import c_parser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils import read_csv, write_pickle

sys.setrecursionlimit(1000000)


def prepared_data(records, max_seq_length=510, test_records=None):
    """
    Main function to read files and prepare the data.
    """
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']

    data, test_data = [], []
    for row in tqdm(records, total=len(data)):
        q2n = []

        sequences = trans_to_sequences(row["tree"])
        sequences = sequences[:max_seq_length]                
        
        tokens_for_sents = ['CLS'] + sequences + ['SEP']
        for token in tokens_for_sents:
            if token not in vocabulary:
                vocabulary[token] = len(inverse_vocabulary)
                q2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(token)
            else:
                q2n.append(vocabulary[token])

        row["q2n"] = q2n
        row["sequences"] = sequences
        row["tokens_for_sents"] = tokens_for_sents
        data.append(row)

    
    if test_records and isinstance(test_records, list):
        for row in tqdm(test_records, total=len(data)):
            q2n = []

            sequences = trans_to_sequences(row["tree"])
            sequences = sequences[:max_seq_length]                
            
            tokens_for_sents = ['CLS'] + sequences + ['SEP']
            for token in tokens_for_sents:
                if token not in vocabulary:
                    vocabulary[token] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(token)
                else:
                    q2n.append(vocabulary[token])

            row["q2n"] = q2n
            row["sequences"] = sequences
            row["tokens_for_sents"] = tokens_for_sents
            test_data.append(row)

    return data, test_data, vocabulary, inverse_vocabulary