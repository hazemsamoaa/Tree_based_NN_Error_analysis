import glob
import os

import numpy as np
import pandas as pd
from extract_ast_paths import Extractor
from models.code2vec.config import Config
from models.code2vec.model_base import Code2VecModelBase
from tqdm.auto import tqdm

from utils import AttrDict, write_pickle

SHOW_TOP_CONTEXTS = 10


def load_model_dynamically(config: Config) -> Code2VecModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}

    if config.DL_FRAMEWORK == 'tensorflow':
        from models.code2vec.tensorflow_model import Code2VecModel
    elif config.DL_FRAMEWORK == 'keras':
        from models.code2vec.keras_model import Code2VecModel

    print(f"Type Code2VecModel: {type(Code2VecModel)}")
    return Code2VecModel(config)


def find_java_files(dir_path):
    java_files = []

    def recursive_search(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                recursive_search(item_path)
            elif item_path.endswith('.java'):
                java_files.append(item_path)

    recursive_search(dir_path)
    return java_files

def parse(config: Config):
    """Parse the tree data from a pickle file and create samples.

    Args:
        args (AttrDict): The arguments containing paths and limits for processing.
    """
    print(f"Loading the model ...")
    model = load_model_dynamically(config)
    print(f"Model is loaded!")

    path_extractor = Extractor(
        jar_path=config.JAR_PATH,
        max_contexts=config.MAX_CONTEXTS,
        max_path_length=config.MAX_PATH_LENGTH,
        max_path_width=config.MAX_PATH_WIDTH
    )
    df = pd.read_csv(f"{config.IN_DIR}/info.csv")
    just_filename = True if "/" not in df.iloc[0]["Test case"] else False

    data = []

    # g_files = glob.glob(f"{config.IN_DIR}/*.java")
    g_files = find_java_files(config.IN_DIR)
    print(f"We have #{len(g_files)} java files")
    for i, java_file in tqdm(enumerate(g_files), total=len(g_files)):
        java_file_test_case = java_file.replace(config.IN_DIR, "").strip()
        # row = df[df["Test case"] == java_file.split("/")[-1]]

        if just_filename:
            row = df[df["Test case"] == java_file_test_case.split("/")[-1]]
        else:
            row = df[df["Test case"] == java_file_test_case]

        if not len(row) > 0:
            print(f"There's something wrong with retrieving data for `{java_file}`!")
            continue

        try:
            lines, hash_to_string_dict = path_extractor.extract_paths(java_file)
        except ValueError as e:
            print(e)
            continue

        if lines:
            representation = model.predict(lines)

            data.append({
                'code_vector': np.vstack([rp.code_vector for rp in representation]),
                'fname': java_file.split("/")[-1],
                'metadata': {'value': row.to_dict(orient="records")[0]["Runtime in ms"]}
            })

    print(f"We have #{len(data)} java files")
    write_pickle(data, os.path.join(config.OUT_DIR, "code2vec.pkl"))
    print('Representation load finished')


if __name__ == '__main__':
    # python ./src/code2vec.py --load ./models/java14_model/saved_model_iter8.release --predict --export_code_vectors --in_dir ../Datasets/OssBuilds/ --out_dir ./data/
    config = Config(set_defaults=True, load_from_args=True, verify=True)
    print(config.__dict__)
    # parse(config)
