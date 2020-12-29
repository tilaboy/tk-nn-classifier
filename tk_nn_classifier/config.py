'''The module to load the config file'''
import os
import copy
import json
from .exceptions import ConfigError

# config field for feature and category
FEAT_FIELD = 'features'
CAT_FIELD = 'class'

# feat types:
FEAT_TYPE_TOKEN = 'token'
FEAT_TYPE_CHAR = 'char'

# default max length
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_MAX_CHARS = 32


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
        "language": "en",
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

# TODO: convert functions to class
class ModelConfig:
    '''
    build:
        - read from json file
        - load from dictionary
    '''
    pass

def _default_feat_property(field_name):
    '''set default feature type as text field'''
    return {'type': FEAT_TYPE_TOKEN, 'max_len': _DEFAULT_MAX_TOKENS}

def _set_default_max_len(property):
    if property['type'] == FEAT_TYPE_TOKEN:
        property['max_len'] = _DEFAULT_MAX_TOKENS
    elif property['type'] == FEAT_TYPE_CHAR:
        property['max_len'] = _DEFAULT_MAX_CHARS

def _check_feat_property(feat_dict):
    for feat_name, property in feat_dict.items():
        if 'type' not in property:
            raise ConfigError(
                'type',
                section=FEAT_FIELD,
                detail_msg='type for {} not set'.format(feat_name)
            )
        if 'max_len' not in property:
            _set_default_max_len(property)

def set_feature_property(cfg_input):
    '''
    set feature property, consider the three case in config:
        - feature: field
        - feature: [field_1, field_2]
        - feature: {'field_1': {'type': 'text', 'max_len': 1024},
                    'field_2': {...}
                    ....
                   }

    internally, all three config type will be convert to type 3

    Note that the dict ordered is maintained since python 3.6, for
    Python versions >=2.7 and <3.6, we will need collections.OrderedDict.
    Since the repo only supports >=3.6, ordereddict is not used here
    '''
    if isinstance(cfg_input[FEAT_FIELD], str):
        field = cfg_input[FEAT_FIELD]
        cfg_input[FEAT_FIELD] = {
            field: _default_feat_property(field)
        }
    elif isinstance(cfg_input[FEAT_FIELD], list):
        cfg_input[FEAT_FIELD] = {
            field: _default_feat_property(field)
            for field in cfg_input[FEAT_FIELD]
        }
    elif isinstance(cfg_input[FEAT_FIELD], dict):
        _check_feat_property(cfg_input[FEAT_FIELD])


def _derived_config_fields(config):
    '''
    modify the config to be used in later data manipulation:
        - model_path = model_dir/model_version
        - model_evel_path = model_path/res
        - dropout_keep_rate = 1 - dropout_rate
        - convert the feature to dictionary format
          field_name: {type: 'token/char', max_len: 111}
    '''
    config['model_path'] = os.path.join(config['model_dir'],
                                        config['model_version'])
    config['model_eval_path'] = os.path.join(config['model_path'],
                                             'res')

    # derived parameters
    config['dropout_keep_rate'] = 1 - config['dropout_rate']

    # convert the config[x][features] to list
    for input_type in ['trxml_fields', 'csv_fields']:
        if input_type in config:
            set_feature_property(config[input_type])


def load_config(config_file: str):
    '''
    load the config file

    params:
        - config_file: the full path of the config (type: string)

    output:
        - dictionary object contains config key and config value pairs
    '''
    with open(config_file) as config_fh:
        config_dikt = json.load(config_fh)
    config_dikt['config_file_path'] = config_file
    return load_config_from_dikt(config_dikt)

def _validate_config(config):
    # obligated terms:
    _validate_must_have(config)
    _validate_at_least_one(config)
    _validate_field_consistency(config)
    if 'spacy' in config:
        _validate_spacy_field_consistency(config)


def _validate_must_have(config):
    fields_must_have = ['model_type', 'model_name',
                        'model_dir', 'model_version',
                        'datasets']

    for field in fields_must_have:
        if field not in config:
            raise ConfigError(field)

    data_fields_must_have = [FEAT_FIELD, CAT_FIELD]

    for doc_type in ['trxml', 'csv']:
        field_entry = doc_type + '_fields'
        if field_entry in config:
            for field in data_fields_must_have:
                if field not in config[field_entry]:
                    raise ConfigError(field, field_entry)


def _validate_at_least_one(config):
    fields_at_least_one = [['trxml_fields', 'csv_fields']]

    for field_set in fields_at_least_one:
        if any(field in config for field in field_set):
            pass
        else:
            raise ConfigError(
                ' or '.join(field_set),
                section='',
                detail_msg='need at least one: {}'.format(', '.join(field_set)))

def _validate_field_consistency(config):
    if 'all_data' in config['datasets']:
        if 'train' in config['datasets'] or 'eval' in config['datasets']:
            raise ConfigError(
                'all_data',
                section='datasets',
                detail_msg='config conflict: all_data <=> train/eval')
    if 'all_data' not in config['datasets'] and 'train' not in config['datasets']:
        raise ConfigError(
            'train',
            'datasets',
            detail_msg='all_data or train need to be set')

def _validate_spacy_field_consistency(config):
    '''
    check the consistency of the language and pretrained model:

    params:
        - config: dict

    output:
        - updated input config or die on error
    '''
    if 'language' not in config['spacy']:
        raise ConfigError('language', 'spacy'
                          'language is needed for spacy model setup')

    language = config['spacy']['language']
    if 'model' in config['spacy']:
        model_name = config['spacy']['model']
        if not model_name.startswith(language):
            detail_msg = 'Spacy model name starts with language iso_code'
            raise ConfigError('model', 'spacy', detail_msg)
    else:
        if language in poc_spacy_lang_model:
            config['spacy']['model'] = poc_spacy_lang_model[language]
        else:
            raise ConfigError('model', 'spacy', f'language {language} not supported')

    if 'arch' not in config['spacy']:
        # default to simple_cnn
        config['spacy']['arch'] = 'simple_cnn'

def data_field_type(field_name, config):
    trxml_type = csv_type = None
    if 'trxml_fields' in config:
        trxml_type = _field_type_lookup(field_name, config['trxml_fields'])
    if 'csv_fields' in config:
        csv_type = _field_type_lookup(field_name, config['csv_fields'])
    if trxml_type and csv_type:
        raise ConfigError(
            field_name, 'trxml_fields/csv_fields',
            f'field name {field_name} in both trxml_fields and csv_fields')
    elif trxml_type:
        field_type = trxml_type
    elif csv_type:
        field_type = csv_type
    else:
        raise ConfigError(
            field_name, 'trxml_fields/csv_fields',
            f'field name {field_name} could not found in trxml_fields or csv_fields')
    return field_type

def _field_type_lookup(field_name, field_config):
    field_type = ''
    for field_type in field_config.keys():
        if isinstance(field_config[field_type], (list, dict)):
            if field_name in field_config[field_type]:
                return field_type
        elif isinstance(field_config[field_type], str):
            if field_name == field_config[field_type]:
                return field_type
        else:
            raise ConfigError(
                field_type, 'trxml_fields/csv_fields',
                f'{field_type} value in un-supported data type')


def load_config_from_dikt(config_dikt):
    '''
    load the config from dikt

    params:
        - config_dikt: config options saved in dictionary

    output:
        - dictionary object contains config key and config value pairs
    '''

    config = get_default_config()
    config.update(config_dikt)
    if not config['model_type'].startswith('spacy'):
        config.pop('spacy', None)
    _derived_config_fields(config)
    _validate_config(config)
    return config
