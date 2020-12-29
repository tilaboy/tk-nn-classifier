from unittest import TestCase
from tk_nn_classifier import config
from tk_nn_classifier.exceptions import ConfigError
from tk_nn_classifier.config import FEAT_TYPE_TOKEN, FEAT_TYPE_CHAR
from tk_nn_classifier.config import FEAT_FIELD, CAT_FIELD
from tk_nn_classifier.config import _DEFAULT_MAX_TOKENS, _DEFAULT_MAX_CHARS

# TODO
# - different config type works:
#   - spacy
#   - keras
#   - tf

class TestDefaultConfig(TestCase):
    '''test the load of config'''
    def setUp(self):
        self.config = config.get_default_config()

    def test_required_model_param(self):
        # type is not empty
        self.assertRegex(self.config['model_type'], r'\w')

    def test_spacy_model_name_consistency(self):
        self.config['spacy']['model'] = 'foo'
        with self.assertRaises(ConfigError):
            config._validate_spacy_field_consistency(self.config)

    def test_spacy_model_not_support_lang(self):
        self.config['spacy']['language'] = 'zh'
        with self.assertRaises(ConfigError):
            config._validate_spacy_field_consistency(self.config)

    def test_set_str_feat_property(self):
        input_feat = {
            "csv_fields": {
                "features": "full_text",
                "class": "advertiser_type",
                "doc_id": "posting_id",
                "extra": ["organization_name", "source_type", "source_website"]
            },
        }
        self.config.update(input_feat)
        config._derived_config_fields(self.config)
        self.assertEqual(
            self.config['csv_fields'][FEAT_FIELD],
            {'full_text': {'type': FEAT_TYPE_TOKEN, 'max_len': _DEFAULT_MAX_TOKENS}},
            'str feat name to dict'
        )

    def test_set_list_feat_property(self):
        input_feat = {
            "csv_fields": {
                "features": ["full_text", "organization_name"],
                "class": "advertiser_type",
                "doc_id": "posting_id",
                "extra": ["organization_name", "source_type", "source_website"]
            },
        }
        self.config.update(input_feat)
        config._derived_config_fields(self.config)
        self.assertEqual(
            self.config['csv_fields'][FEAT_FIELD],
            {'full_text': {'type': FEAT_TYPE_TOKEN, 'max_len': _DEFAULT_MAX_TOKENS},
             'organization_name': {'type': FEAT_TYPE_TOKEN, 'max_len': _DEFAULT_MAX_TOKENS},
             },
            'list of feat name to dict'
        )

    def test_check_dict_feat_property(self):
        input_feat = {
            "csv_fields": {
                "features": {
                    "full_text": {'type': FEAT_TYPE_TOKEN},
                    "organization_name": {'type': FEAT_TYPE_CHAR}
                },
                "class": "advertiser_type",
                "doc_id": "posting_id",
                "extra": ["organization_name", "source_type", "source_website"]
            },
        }
        self.config.update(input_feat)
        config._derived_config_fields(self.config)
        self.assertEqual(
            self.config['csv_fields'][FEAT_FIELD],
            {'full_text': {'type': FEAT_TYPE_TOKEN, 'max_len': _DEFAULT_MAX_TOKENS},
             'organization_name': {'type': FEAT_TYPE_CHAR, 'max_len': _DEFAULT_MAX_CHARS},
             },
            'dict feat name to dict'
        )

    def test_check_complete_dict_feat_property(self):
        input_feat = {
            "csv_fields": {
                "features": {
                    "full_text": {'type': FEAT_TYPE_TOKEN, 'max_len': 512},
                    "organization_name": {'type': FEAT_TYPE_CHAR, 'max_len': 64}
                },
                "class": "advertiser_type",
                "doc_id": "posting_id",
                "extra": ["organization_name", "source_type", "source_website"]
            },
        }
        self.config.update(input_feat)
        config._derived_config_fields(self.config)
        self.assertEqual(
            self.config['csv_fields'][FEAT_FIELD],
            {'full_text': {'type': FEAT_TYPE_TOKEN, 'max_len': 512},
             'organization_name': {'type': FEAT_TYPE_CHAR, 'max_len': 64},
             },
            'complete dict feat name to dict'
        )
