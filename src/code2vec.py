import glob

from tqdm.auto import tqdm

from extract_ast_paths import Extractor
from models.code2vec.config import Config
from models.code2vec.model_base import Code2VecModelBase
from utils import AttrDict


def load_model_dynamically(config: Config) -> Code2VecModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}

    if config.DL_FRAMEWORK == 'tensorflow':
        from models.code2vec.tensorflow_model import Code2VecModel
    elif config.DL_FRAMEWORK == 'keras':
        from models.code2vec.keras_model import Code2VecModel

    print(f"Type Code2VecModel: {type(Code2VecModel)}")
    return Code2VecModel(config)


def parse(config: Config):
    """Parse the tree data from a pickle file and create samples.

    Args:
        args (AttrDict): The arguments containing paths and limits for processing.
    """
    model = load_model_dynamically(config)

    path_extractor = Extractor(
        jar_path=config.JAR_PATH,
        max_contexts=config.MAX_CONTEXTS,
        max_path_length=config.MAX_PATH_LENGTH,
        max_path_width=config.MAX_PATH_WIDTH
    )

    vector_representations = []
    for java_file in tqdm(glob.glob(f"{config.IN_DIR}/*.java")):

        try:
            lines, hash_to_string_dict = path_extractor.extract_paths(java_file)
        except ValueError as e:
            print(e)
            continue

        representation = model.predict(lines)
        vector_representations.append(representation)

    print('Representation load finished')


if __name__ == '__main__':
    # python src/code2vec.py --load ./data/models/java14_model/saved_model_iter8.release --predict --export_code_vectors --in_dir ./data/java/rdf4j_codes/
    config = Config(set_defaults=True, load_from_args=True, verify=True)
    parse(config)
