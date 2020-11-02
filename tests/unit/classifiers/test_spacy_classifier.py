'''test: mutle feature input got parsed correctly as model input'''
import os
from unittest import TestCase
import tempfile
import shutil
import json
from spacy.lang.en import English
from tk_nn_classifier.classifiers import SpacyClassifier
from tk_nn_classifier.classifiers.utils import eval_predictions
from tk_nn_classifier.config import load_config_from_dikt


class SpacyClassifierTestCases(TestCase):
    '''unit test for spacy classifier:
        - data preparation
        - model Training
        - loading
        - evaluation'''

    @classmethod
    def setUpClass(self):
        self.test_train = 'tests/resource/sample_train.csv'
        self.test_eval = 'tests/resource/sample_eval.csv'
        self.test_trxml = 'tests/resource/samples/'
        self.test_dir = tempfile.mkdtemp()
        config_dikt = {
            "model_type": "spacy_simple",
            "model_name": "spacy_simple",
            "model_dir": self.test_dir,
            "model_version": "test",

            "dropout_rate": 0.5,
            "num_epochs": 8,
            "max_lines":50,

            "spacy": {
                "model": "en_core_web_sm",
                "lang": "en",
                "arch": "simple_cnn"
            },

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
                "extra": ["organization_name", "source_url"]
            },

            "datasets": {
                "train": self.test_train,
                "eval": self.test_eval,
                "test": {
                    "test": self.test_trxml
                },
                "label_mapper": self.test_dir + "label_mapper.json"
            }
        }
        self.config = load_config_from_dikt(config_dikt)
        #self.data_reader = DataReader(self.config)

    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_01_prepare_data(self):
        classifier = SpacyClassifier(self.config)

        self.assertEqual(self.test_train, classifier.config['datasets']['train'])
        self.assertEqual(self.test_eval, classifier.config['datasets']['eval'])

        train_data, eval_data = classifier.prepare_data()

        # train data
        self.assertEqual(len(train_data), 256)
        self.assertEqual(len(train_data[6][0]), 2575, 'check train doc 6 length')
        self.assertEqual(train_data[6][1],
                         {'cats': {'no': True, 'yes': False}},
                         'check train doc 6 type')

        # eval data
        self.assertEqual(len(eval_data), 64)
        self.assertEqual(len(eval_data[6][0]), 3698, 'check eval doc 6 length')
        self.assertEqual(eval_data[6][1], {'no': True, 'yes': False})

        # also check the label id
        with open(classifier.config['datasets']['label_mapper']) as mapper_fh:
            label_mapper = json.load(mapper_fh)
        self.assertEqual(label_mapper, {"0": "no", "1": "yes"})

    def test_02_build_graph(self):
        classifier = SpacyClassifier(self.config)
        classifier.build_graph()
        self.assertTrue(isinstance(classifier.model, English))
        self.assertTrue(classifier.model.has_pipe('textcat'))

    def test_03_train_save_and_eval(self):
        classifier = SpacyClassifier(self.config)
        train_data, eval_data = classifier.prepare_data()
        classifier.build_graph()
        classifier.train(train_data, eval_data)
        classifier.save(classifier.config['model_path'])
        eval, gold = classifier.evaluate(eval_data)
        accuracy, prediction, recall = eval_predictions(eval, gold)
        self.assertGreater(accuracy, 0.8, 'testing on eval set using trained model')

    def test_04_load_and_eval(self):
        classifier = SpacyClassifier(self.config)
        test_set = classifier.data_reader.get_data(classifier.config['datasets']['test']['test'])
        classifier.load_saved_model()
        eval, gold = classifier.evaluate(test_set, mode='test')
        accuracy, precision, recall = eval_predictions(eval, gold)
        self.assertGreater(accuracy, 0.6, 'testing on test set using trained model')
