'''test: mutle feature input got parsed correctly as model input'''
import os
from unittest import TestCase
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tk_nn_classifier.classifiers import KerasClassifier
from tk_nn_classifier.config import load_config_from_dikt
from tk_nn_classifier.classifiers.utils import eval_predictions

class KerasClassifierTestCases(TestCase):
    '''unit test for tensorflow classifier:
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
            "model_type": "keras_cnn_multi",
            "model_name": "keras_simple",
            "model_dir": self.test_dir,
            "model_version": "test_01",
            "language": "en",
            "log_dir": "log",

            "dropout_rate": 0.5,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "num_epochs": 500,
            "batch_size": 128,
            "max_lines":50,

            "log_dir": "log",
            "max_sequence_length": 1024,
            "patience_epochs": 10,

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
            "lstm": {
                "hidden_size": 150,
                "nr_layers": 2
            },

            "cnn": {
                "nr_layers": 3,
                "filter_size": 32,
                "kernel_size": 3
            },
            "embedding": {
                "filepath": "tests/resource/sample_embedding.bin",
                "dimension": 150,
                "token_encoding": "max_embedding",
                "trainable": False,
                "use_local": True
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
        #self.data_reader = DataReader(self.config)
        self.config = load_config_from_dikt(config_dikt)

    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_00_load_embedding(self):
        classifier = KerasClassifier(self.config)
        classifier.load_embedding()
        self.assertEqual(classifier.embedding.vocab_size, 8854)
        self.assertEqual(classifier.embedding.vector_size, 150)

    def test_01_prepare_data(self):
        classifier = KerasClassifier(self.config)
        classifier.load_embedding()
        (train_data, labels, train_data_length) = classifier.load_data_set(
            classifier.config['datasets']['train'])

        # index of two tokens from embedding
        self.assertEqual(train_data.shape, (256, 1024, 150))
        self.assertEqual(train_data_length[0], 551)
        self.assertEqual(sum(labels), 109)


        # also check the label id
        # with open(classifier.config['datasets']['label_mapper']) as mapper_fh:
        #    label_mapper = json.load(mapper_fh)
        # self.assertEqual(label_mapper, {"0": "no", "1": "yes"})

    def test_02_build_graph(self):
        classifier = KerasClassifier(self.config)
        classifier.load_embedding()
        classifier.build_graph()
        self.assertTrue(isinstance(classifier.classifier, Model))

    def test_03_train_save_and_eval(self):
        classifier = KerasClassifier(self.config)
        classifier.load_embedding()
        train_data, eval_data = classifier.prepare_train_eval_data()
        classifier.build_graph()
        classifier.train(train_data, eval_data)

        likelihoods, gold = classifier.evaluate(classifier.config['datasets']['eval'])
        eval = [1 if likelihood >= 0.5 else 0 for likelihood in likelihoods]
        accuracy, precision, recall = eval_predictions(eval, gold)
        self.assertGreater(accuracy, 0.7, 'testing on eval set using trained model')

    #def test_04_load_and_eval(self):
    #    classifier = SpacyClassifier(self.config)
    #    test_set = classifier.data_reader.get_data(classifier.config['datasets']['test']['test'])
    #    classifier.load_saved_model()
    #    eval, gold = classifier.evaluate(test_set, mode='test')
    #    accuracy, prediction, recall = eval_predictions(eval, gold)
    #    self.assertGreater(accuracy, 0.6, 'testing on test set using trained model')
