'''test: mutle feature input got parsed correctly as model input'''
import os
from unittest import TestCase
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from tk_nn_classifier.classifiers import TFClassifier
from tk_nn_classifier.config import load_config_from_dikt

tf.enable_eager_execution()

class TFClassifierTestUtilsCases(TestCase):
    '''unit test for tensorflow classifier utils
        - data preparation
        - padding
    '''

    @classmethod
    def setUpClass(self):
        self.test_dir = tempfile.mkdtemp()
        test_embedding_file = os.path.join(self.test_dir, 'test_embedding.txt')
        with open(test_embedding_file, 'w') as embedding_fh:
            embedding_fh.write(self.embedding_content())

        config_dikt = {
            "model_type": "tf_cnn_simple",
            "model_name": "tf_simple",
            "model_dir": self.test_dir,
            "model_version": "test",

            "max_lines": 5,
            "max_sequence_length": 4,
            "trxml_fields": {
                "features": "sec_vacancy.0.sec_vacancy",
                "class": "derived_vac_intermediary.0.derived_vac_intermediary",
                "doc_id": "Document.0.correlationid",
                "extra": ["derived_org_name.0.derived_org_name",
                    "derived_norm_url.0.derived_norm_url"]
            },
            "csv_fields": {
                "features": "full_text",
                "class": "advertiser_type",
                "doc_id": "posting_id",
                "extra": ["advertiser_name", "source_url"]
            },
            "embedding": {
                "filepath": test_embedding_file,
                "dimension": 3,
                "token_encoding": "token_embedding",
                "trainable": False,
                "use_local": True

            },

            "datasets": {
                'train': 'foo',
                'eval': 'bar'
            }

        }
        #self.data_reader = DataReader(self.config)
        config = load_config_from_dikt(config_dikt)
        self.classifier = TFClassifier(config)
        self.cases = [
            {
                "input": 'foo bar zoo new',
                "ids": [2, 3, 4, 6],
                "length": 4
            },
            {
                "input": 'foo bar zoo boo foo new rule',
                "ids": [2, 3, 4, 5],
                "length": 4
            },
            {
                "input": 'bar zoo',
                "ids": [3, 4, 0, 0],
                "length": 2
            },
            {
                "input": 'foo bar zoo boo foo',
                "ids": [2, 3, 4, 5],
                "length": 4
            },
            {
                "input": 'new rule',
                "ids": [6, 8, 0, 0],
                "length": 2
            },
        ]

    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_prepare_single_input(self):
        self.classifier.load_embedding()
        for case in self.cases:
            data = self.classifier._prepare_single_input(case['input'])
            self.assertEqual(data[0]['input'].numpy().tolist(), case['ids'])
            self.assertEqual(data[0]['len'].numpy(), case['length'])

    def test_input_text_to_pad_id(self):
        self.classifier._load_vocab()
        for case in self.cases:
            data = self.classifier._input_text_to_pad_id(case['input'])
            self.assertEqual(data["input"][0].tolist(), case['ids'])


    @staticmethod
    def embedding_content():
        embedding_content = '''7 3
FOO 0.1 1.0 -0.3
BAR 1.0 -0.6 1.0
ZOO -0.5 0.5 0.1
BOO -0.7 -0.7 0.6
NEW 0.2 -1.0 0.7
OLD -1.0 0.9 -1.0
RULE 0.7 -0.1 0.0
'''
        return embedding_content
