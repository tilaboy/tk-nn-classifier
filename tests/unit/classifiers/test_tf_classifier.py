'''test: mutle feature input got parsed correctly as model input'''
import os
from unittest import TestCase
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from tk_nn_classifier.classifiers import TFClassifier
tf.enable_eager_execution()

class TFClassifierTestCases(TestCase):
    '''unit test for multi feature input and parsing'''

    def setUp(self):
        self.trxml_dir = 'tests/resource/samples'
        self.csv_file = 'tests/resource/us_small.csv'
        self.test_dir = tempfile.mkdtemp()
        test_embedding_file = os.path.join(self.test_dir, 'test_embedding.txt')
        with open(test_embedding_file, 'w') as embedding_fh:
            embedding_fh.write(self.embedding_content())

        config= {
            "model_type": "tf_cnn_simple",
            "max_lines": 5,
            "model_path": self.test_dir,
            "max_sequence_length": 4,
            "trxml_fields": {
                "features": "sec_vacancy.0.sec_vacancy",
                "class": "derived_vac_intermediary.0.derived_vac_intermediary",
                "doc_id": "Document.0.correlationid",
                "extra": ["derived_org_name.0.derived_org_name",
                    "derived_source_site.0.derived_source_site",
                    "derived_norm_url.0.derived_norm_url"]
            },
            "csv_fields": {
                "features": "full_text",
                "class": "source_type",
                "doc_id": "posting_id",
                "extra": ["advertiser_name", "source_website", "source_url"]
            },
            "embedding": {
                "file": test_embedding_file,
                "dimension": 3,
                "token_encoding": "token_embedding",
                "trainable": False
            },

            "datasets": {}

        }
        #self.data_reader = DataReader(self.config)
        self.classifier = TFClassifier(config)
        self.classifier.load_embedding()

    def tearDown(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_input_text_to_pad_id(self):
        inputs = [
                    'foo bar zoo new',
                    'foo bar zoo boo foo new rule',
                    'boo zoo'
                 ]

        expected_ids = [ [2, 3, 4, 6], [2, 3, 4, 5], [5, 4, 0, 0] ]
        expected_lengths = [ 4, 4, 2 ]

        for (input, expected_id, expected_length) in zip(inputs, expected_ids, expected_lengths):
            transfered_input = self.classifier._prepare_single_input(input)
            self.assertEqual(transfered_input[0]['input'].numpy().tolist(), expected_id)
            self.assertEqual(transfered_input[0]['len'].numpy(), expected_length)

    def test_input_text_to_pad_id(self):
        test_texts = ['foo bar zoo boo foo', 'new rule']
        expected_ids = [[2, 3, 4, 5], [6, 8, 0, 0]]
        self.classifier._load_vocab()
        for test_text, expected_id in zip(test_texts, expected_ids):
            data = self.classifier._input_text_to_pad_id(test_text)
            self.assertEqual(data["input"].tolist(), [expected_id] )


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
