from unittest import TestCase
from tk_nn_classifier import config
from tk_nn_classifier.exceptions import ConfigError

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
            config.spacy_lang_model_consistency(self.config)

    def test_spacy_model_not_support_lang(self):
        self.config['spacy']['lang'] = 'zh'
        with self.assertRaises(ConfigError):
            config.spacy_lang_model_consistency(self.config)
