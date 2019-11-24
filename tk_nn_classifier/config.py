'''The module to load the config file'''
import copy
import json

DEFAULTS = {
    # Data reading params
    "model_type": 'spacy_poc',
    "model_name": "poc_model",
    "model_path": "models/poc",

    "split_ratio": 0.8,
    "dropout_rate": 0.2,
    "num_epochs": 20,
    "max_lines":50,

    "spacy": {
        "lang": "en",
        "arch": "simple_cnn"
    },

    # Embedding
    "embedding": {},
    "cnn": {},

    # Training data files
    "datasets": {},
}

poc_spacy_lang_model = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm',
    'nl': 'nl_core_news_sm',
    'es': 'es_core_news_sm',
    'pt': 'pt_core_news_sm',
    'it': 'it_core_news_sm'
}


def get_default_config():
    '''
    get the default config
    '''
    config = copy.deepcopy(DEFAULTS)
    return config


def load_config(config_file, poc_defaults=False):
    '''
    load the config file

    params:
        - config_file: the full path of the config (type: string)

    output:
        - dictionary object contains config key and config value pairs
    '''

    if poc_defaults:
        config = get_default_config()
        language = config['spacy']['lang']
        config['spacy']['model'] = poc_spacy_lang_model[language]
    else:
        config = {}
    config['config_file_path'] = config_file
    with open(config_file) as config_fh:
        config.update(json.load(config_fh))
    return config
