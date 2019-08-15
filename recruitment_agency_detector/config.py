'''The module to load the config file'''
import copy
import json

DEFAULTS = {
    # Data reading params
    "features_enabled": True,
    "token_encoding": "word_embedding",

    # Training config
    "optimizer": "Adam",
    "learning_rate": 0.002,
    "num_epochs": 100,
    "batch_size": 64,

    # Where to save the model
    "log_dir": None,
    "model_name": None,

    # Embedding
    "embedding_file": None,

    # Training data files
    "train_data": None,
    "test_sets": {},
}


def get_default_config():
    '''
    get the default config
    '''
    config = copy.deepcopy(DEFAULTS)
    return config


def load_config(config_file, insert_defaults=False):
    '''
    load the config file

    params:
        - config_file: the full path of the config (type: string)

    output:
        - dictionary object contains config key and config value pairs
    '''

    if insert_defaults:
        config = get_default_config()
    else:
        config = {}
    with open(config_file) as config_fh:
        config.update(json.load(config_fh))
    return config
