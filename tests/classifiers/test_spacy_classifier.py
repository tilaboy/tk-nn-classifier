'''test: mutle feature input got parsed correctly as model input'''
import os
from unittest import TestCase
import tempfile
import shutil
import json
from spacy.lang.en import English
from tk_nn_classifier.classifiers import SpacyClassifier
from tk_nn_classifier.classifiers.utils import eval_accuracy
from tk_nn_classifier.config import load_config_from_dikt
from tk_nn_classifier.data_loader import load_data_set

train_data_path = 'tests/resource/train.csv'
eval_data_path = 'tests/resource/eval.csv'
test_data_path = 'tests/resource/sample_trxmls/'

config_dikt = {
    "model_type": "spacy_simple",
    "model_name": "spacy_simple",
    "model_dir": '',
    "model_version": "test",

    "dropout_rate": 0.5,
    "num_epochs": 20,
    "max_lines":50,

    "spacy": {
        "model": "en_core_web_sm",
        "language": "en",
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
        "train": train_data_path,
        "eval": eval_data_path,
        "test": {
            "test": test_data_path
        },
        "label_mapper": ''
    }
}


def _load_train_eval(classifier):
    train_docs = load_data_set(
        classifier.config,
        classifier.config['datasets']['train'],
        train_mode=True)
    eval_docs = load_data_set(
        classifier.config,
        classifier.config['datasets']['eval'],
        train_mode=False)
    train_data = classifier.data_reader.model_input(train_docs, train_mode=True)
    eval_data = classifier.data_reader.model_input(eval_docs, train_mode=False)
    return train_data, eval_data

def _load_mapper(mapper_file):
    with open(mapper_file) as mapper_fh:
        label_mapper = json.load(mapper_fh)
    return label_mapper


class SpacyClassifierTestCases(TestCase):
    '''unit test for spacy classifier:
        - data preparation
        - model Training
        - loading
        - evaluation'''

    @classmethod
    def setUpClass(self):
        self.test_dir = tempfile.mkdtemp()
        config_dikt['model_dir'] = self.test_dir
        config_dikt['datasets']['label_mapper'] = os.path.join(self.test_dir, 'label_mapper.json')
        self.config = load_config_from_dikt(config_dikt)


    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_00_config(self):
        self.assertEqual(self.config['trxml_fields']['features'], ["sec_vacancy.0.sec_vacancy"])
        self.assertEqual(self.config['csv_fields']['features'], ["full_text"])
        self.assertEqual(
            self.config['model_path'],
            os.path.join(self.config['model_dir'], self.config['model_version'])
        )

    def test_01_prepare_data(self):
        classifier = SpacyClassifier(self.config)
        train_data, eval_data = _load_train_eval(classifier)
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
        label_mapper = _load_mapper(self.config['datasets']['label_mapper'])
        self.assertEqual(label_mapper, {"0": "no", "1": "yes"})

    def test_02_build_graph(self):
        classifier = SpacyClassifier(self.config)
        classifier.build_graph()
        self.assertTrue(isinstance(classifier.model, English))
        self.assertTrue(classifier.model.has_pipe('textcat'))

    def test_03_train_save_eval(self):
        classifier = SpacyClassifier(self.config)
        train_data, eval_data = _load_train_eval(classifier)
        classifier.build_graph()
        classifier.train(train_data, eval_data)
        classifier.save(classifier.config['model_path'])

        eval, gold = classifier.eval_test_set(self.config['datasets']['eval'])
        accuracy = eval_accuracy(eval, gold)
        self.assertGreater(accuracy, 0.8, 'testing on eval set using trained model')

    def test_04_load_test(self):
        classifier = SpacyClassifier(self.config)
        classifier.load_saved_model(classifier.config['model_path'])
        test_file = classifier.config['datasets']['test']['test']
        eval, gold = classifier.eval_test_set(test_file)
        accuracy = eval_accuracy(eval, gold)
        self.assertGreater(accuracy, 0.8, 'testing on test set using trained model')


class SpacyClassifierMultiFieldsTestCases(TestCase):
    '''unit test for spacy classifier with multi field input'''
    @classmethod
    def setUpClass(self):
        self.test_dir = tempfile.mkdtemp()
        config_dikt['model_dir'] = self.test_dir
        config_dikt['datasets']['label_mapper'] = os.path.join(self.test_dir, 'label_mapper.json')
        config_dikt['csv_fields']['features'] = ['organization_name', 'full_text']
        config_dikt['trxml_fields']['features'] = [
            'derived_org_name.0.derived_org_name',
            'sec_vacancy.0.sec_vacancy'
        ]
        config_dikt['model_version'] = 'test_multi_fields'
        self.config = load_config_from_dikt(config_dikt)

    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_00_config_multi_fields(self):
        self.assertEqual(
            self.config['trxml_fields']['features'],
            ['derived_org_name.0.derived_org_name', 'sec_vacancy.0.sec_vacancy']
        )
        self.assertEqual(
            self.config['csv_fields']['features'],
            ['organization_name', 'full_text']
        )
        self.assertEqual(
            self.config['model_path'],
            os.path.join(self.config['model_dir'], self.config['model_version'])
        )

    def test_01_prepare_data_multi_fields(self):
        classifier = SpacyClassifier(self.config)
        train_data, eval_data = _load_train_eval(classifier)
        # train data
        self.assertEqual(len(train_data), 256)
        self.assertEqual(len(train_data[6][0]), 2599, 'check train doc 6 length')
        self.assertEqual(train_data[6][1],
                         {'cats': {'no': True, 'yes': False}},
                         'check train doc 6 type')

        # eval data
        self.assertEqual(len(eval_data), 64)
        self.assertEqual(len(eval_data[6][0]), 3716, 'check eval doc 6 length')
        self.assertEqual(eval_data[6][1], {'no': True, 'yes': False})

        # also check the label id
        label_mapper = _load_mapper(self.config['datasets']['label_mapper'])
        self.assertEqual(label_mapper, {"0": "no", "1": "yes"})


    def test_03_train_save_eval_multi_fields(self):
        classifier = SpacyClassifier(self.config)
        train_data, eval_data = _load_train_eval(classifier)
        classifier.build_graph()
        classifier.train(train_data, eval_data)
        classifier.save(classifier.config['model_path'])
        eval, gold = classifier.eval_test_set(classifier.config['datasets']['eval'])
        accuracy = eval_accuracy(eval, gold)
        self.assertGreater(accuracy, 0.8, 'testing on eval set using trained model')

    def test_04_load_test_multi_fields(self):
        classifier = SpacyClassifier(self.config)
        classifier.load_saved_model(classifier.config['model_path'])
        test_file = classifier.config['datasets']['test']['test']
        eval, gold = classifier.eval_test_set(test_file)
        accuracy = eval_accuracy(eval, gold)
        self.assertGreater(accuracy, 0.7, 'testing on test set using trained model')


class SpacyClassifierMultiLabelTestCases(TestCase):
    '''unit test for spacy classifier with multi label'''
    @classmethod
    def setUpClass(self):
        self.test_dir = tempfile.mkdtemp()
        config_dikt['model_dir'] = self.test_dir
        config_dikt['datasets'] = {
            'train': test_data_path,
            'eval': test_data_path,
            'test': {
                'test': test_data_path
            },
            'label_mapper': os.path.join(self.test_dir, 'label_mapper.json')
        }
        config_dikt.pop('csv_fields', None)
        config_dikt['trxml_fields']['features'] ='sec_vacancy.0.sec_vacancy'
        config_dikt['trxml_fields']['class'] = 'derived_cond_contract_type.0.derived_cond_contract_type'
        config_dikt['model_version'] = 'test_multi_label'
        self.config = load_config_from_dikt(config_dikt)

    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_00_config_multi_labels(self):
        self.assertEqual(
            self.config['trxml_fields']['features'],
            ['sec_vacancy.0.sec_vacancy']
        )
        self.assertEqual(
            self.config['trxml_fields']['class'],
            'derived_cond_contract_type.0.derived_cond_contract_type'
        )
        self.assertEqual(
            self.config['model_path'],
            os.path.join(self.config['model_dir'], self.config['model_version'])
        )

    def test_01_prepare_data_multi_labels(self):
        classifier = SpacyClassifier(self.config)
        train_data, eval_data = _load_train_eval(classifier)

        # train data
        self.assertEqual(len(train_data), 64)
        self.assertEqual(len(train_data[6][0]), 1699, 'check train doc 6 length')
        self.assertEqual(train_data[6][1],
                         {'cats': {'Detachering / interim': False,
                                   'Franchise': False,
                                   'Freelance': False,
                                   'Mogelijk vast': False,
                                   'Tijdelijk': True,
                                   'Unspecified': False,
                                   'Vast': False
                                   }
                          },
                         'check train doc 6 type')

        # eval data is the same as train

        # also check the label id
        label_mapper = _load_mapper(self.config['datasets']['label_mapper'])
        self.assertEqual(
            label_mapper,
            {'0': 'Detachering / interim', '1': 'Franchise',
             '2': 'Freelance', '3': 'Mogelijk vast', '4': 'Tijdelijk',
             '5': 'Unspecified', '6': 'Vast'}
        )

    def test_03_train_save_eval_multi_labels(self):
        classifier = SpacyClassifier(self.config)
        train_data, eval_data = _load_train_eval(classifier)
        classifier.build_graph()
        classifier.train(train_data, eval_data)
        classifier.save(classifier.config['model_path'])
        eval, gold = classifier.eval_test_set(classifier.config['datasets']['eval'])
        accuracy = eval_accuracy(eval, gold)
        self.assertGreater(accuracy, 0.8, 'testing on eval set using trained model')
        self.assertEqual(
            eval[:10],
            ['Unspecified', 'Vast', 'Unspecified', 'Unspecified', 'Unspecified',
             'Unspecified', 'Tijdelijk', 'Unspecified', 'Vast', 'Unspecified'])


    def test_04_load_eval_multi_labels(self):
        classifier = SpacyClassifier(self.config)
        classifier.load_saved_model(classifier.config['model_path'])
        test_file = classifier.config['datasets']['test']['test']
        eval, gold = classifier.eval_test_set(test_file)
        accuracy = eval_accuracy(eval, gold)
        self.assertGreater(accuracy, 0.8, 'testing on test set using trained model')
