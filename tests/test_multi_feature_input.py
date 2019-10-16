'''test: mutle feature input got parsed correctly as model input'''
import os
from unittest import TestCase
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from recruitment_agency_detector.classifiers import TFMultiFeatClassifier
tf.enable_eager_execution()

class MultiFeatTestCases(TestCase):
    '''unit test for multi feature input and parsing'''

    def setUp(self):
        self.trxml_dir = 'tests/resource/samples'
        self.csv_file = 'tests/resource/us_small.csv'
        self.test_dir = tempfile.mkdtemp()
        test_embedding_file = os.path.join(self.test_dir, 'test_embedding.txt')
        with open(test_embedding_file, 'w') as embedding_fh:
            embedding_fh.write(self.embedding_content())

        config= {
            "model_type": "tf_multi_feat_2",
            "max_lines": 5,
            "model_path": 'foo',
            "max_sequence_length": [4, 2],
            "trxml_fields": {
                "features": ["sec_vacancy.0.sec_vacancy", 'derived_org_name.0.derived_org_name'],
                "class": "derived_vac_intermediary.0.derived_vac_intermediary",
                "doc_id": "Document.0.correlationid",
                "extra": ["derived_org_name.0.derived_org_name",
                    "derived_source_site.0.derived_source_site",
                    "derived_norm_url.0.derived_norm_url"]
            },
            "csv_fields": {
                "features": ["full_text", 'advertiser_name'],

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
        self.classifier = TFMultiFeatClassifier(config)
        self.classifier.load_embedding()

    def tearDown(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_text_to_tokenid(self):
        inputs = [
                    ['foo bar zoo', 'new'],
                    ['foo bar zoo boo foo', 'new rule'],
                    ['boo zoo', 'old rule new']
                 ]

        expected_id = [
            [ [2, 3, 4, 0], [6, 0] ],
            [ [2, 3, 4, 5], [6, 8] ],
            [ [5, 4, 0, 0], [7, 8] ]
        ]
        expected_length = [
            [3, 1], [4, 2], [2, 2]
        ]

        (transfered_id, transfered_length) = self.classifier._inputs_to_features(inputs)
        # TODO: seems not compare anything, manually checked here
        #print(transfered_id)
        #print(transfered_length)
        np.array_equal(transfered_id, expected_id)
        np.array_equal(transfered_length, expected_length)

    def test_data_to_tf_input(self):
        def _data_parser(length, label, *inputs):
            features = {"len": length}
            for index, input in enumerate(inputs):
                features["input_" + str(index)] = input
            return features, label

        inputs = [
                    ['foo bar zoo', 'new'],
                    ['foo bar zoo boo foo', 'new rule'],
                    ['boo zoo', 'old rule new']
                 ]
        labels = [1,0,1]
        (data, data_length) = self.classifier._inputs_to_features(inputs)
        ds = tf.data.Dataset.from_tensor_slices((data_length, labels, *data))
        print(ds.output_types, end=' : ')
        print(ds.output_shapes)
        for e in ds:
            print (e)

        ds = ds.map(_data_parser)
        iterator = ds.make_one_shot_iterator()

        i = 0
        for item in iterator:
            #self.assertEqual(item["input_0"], data[0][i])
            print(item)
            i += 1

        self.assertEqual(len(ds), 5)

    #def test_input_to_features(self):


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
