'''The module to load the config file'''
import os
import copy
import json

DEFAULTS = {
    # Data reading params
    "model_type": 'spacy',
    "model_name": "spacy_poc",
    "model_dir": "models/poc",
    "model_version": "baseline",

    "dropout_rate": 0.2,
    "num_epochs": 20,
    "max_lines": 50,

    "spacy": {
        "lang": "en",
        "arch": "simple_cnn"
    },

    # split ratio for dataset
    "split_ratio": 0.8
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


def spacy_lang_model_consistency(config):
    '''
    check the consistency of the language and pretrained model:

    params:
        - config: dict

    output:
        - updated input config or die on error
    '''

    language = config['spacy']['lang']
    if 'model' in config['spacy']:
        assert config['spacy']['model'].startswith(language), '''Config error:
spacy model should start with the language iso_code'''
    else:
        if language in poc_spacy_lang_model:
            config['spacy']['model'] = poc_spacy_lang_model[language]
        else:
            raise ValueError('Sorry: %s is not supported by spaCy')

    if 'arch' not in config['spacy']:
        # default to simple_cnn
        config['spacy']['arch'] = 'simple_cnn'


def load_config(config_file, poc_defaults=False):
    '''
    load the config file

    params:
        - config_file: the full path of the config (type: string)

    output:
        - dictionary object contains config key and config value pairs
    '''

    config = get_default_config()
    config['config_file_path'] = config_file
    with open(config_file) as config_fh:
        config.update(json.load(config_fh))

    if config['model_type'].startswith('spacy'):
        config['classifier_frame'] = 'SpacyClassifier'
    else:
        config['spacy'] = None
        if config['model_type'].startswith('tf_multi_feat'):
            config['classifier_frame'] = 'TFMultiFeatClassifier'
        elif config['model_type'].startswith('tf'):
            config['classifier_frame'] = 'TFMultiFeatClassifier'
        elif config['model_type'].startswith('keras'):
            config['classifier_frame'] = 'KerasClassifier'
        else:
            raise ValueError("unknown model type [{}]".format(config['model_type']))

    return config
