from unittest import TestCase
import os
import tempfile
import shutil

from tk_nn_classifier.data_loader import load_data_set, _select_data_reader, split_data_set, _data_type
from tk_nn_classifier.data_loader.csv_loader import CSVLoader
from tk_nn_classifier.data_loader.trxml_loader import TRXMLLoader
from tk_nn_classifier.exceptions import FileTypeError

class DataLoaderTestCase(TestCase):
    @classmethod
    def setUpClass(self):
        self.config= {
            "csv_fields": {
                "features": ["full_text", 'advertiser_name'],
                "class": "source_type",
                "doc_id": "posting_id",
                "extra": ["advertiser_name", "source_website", "source_url"]
            },
            "trxml_fields": {
                "features": ["sec_vacancy.0.sec_vacancy", 'derived_org_name.0.derived_org_name'],
                "class": "derived_vac_intermediary.0.derived_vac_intermediary",
                "doc_id": "Document.0.correlationid",
                "extra": ["derived_org_name.0.derived_org_name",
                    "derived_source_site.0.derived_source_site",
                    "derived_norm_url.0.derived_norm_url"]
            }
        }

        self.test_dir = tempfile.mkdtemp()
        self.csv_file = 'tests/resource/sample.csv'
        self.trxml_dir = 'tests/resource/sample_trxmls'

    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_select_data_reader(self):
        data_reader = _select_data_reader(self.config['csv_fields'],
                                          _data_type('tests/resource/sample.csv'))

        self.assertIsInstance(data_reader, CSVLoader)

        data_reader = _select_data_reader(self.config['trxml_fields'],
                                          _data_type('tests/resource/sample_trxmls'))
        self.assertIsInstance(data_reader, TRXMLLoader)

        with self.assertRaisesRegex(FileNotFoundError, 'not found') as cm:
            _data_type('tests/resource/no_exist.csv')

        with self.assertRaisesRegex(FileNotFoundError, 'not found') as cm:
            _data_type('tests/resource/no_exist')

        with self.assertRaisesRegex(FileTypeError, 'not support') as cm:
            _data_type('tests/resource/label_mapper.json')

    def test_load_csv_data_set(self):
        csv_samples = list(load_data_set(self.config, self.csv_file))
        self.assertEqual(len(csv_samples), 30)
        self.assertEqual(list(csv_samples[0].keys()),
                         ['full_text','advertiser_name', 'source_type'])
        csv_samples = list(load_data_set(self.config, self.csv_file, train_mode=False))
        self.assertEqual(len(csv_samples), 30)
        self.assertEqual(list(csv_samples[0].keys()),
                         ['full_text','advertiser_name', 'source_type',
                          'posting_id', 'source_website', 'source_url'])

    def test_load_trxml_data_set(self):
        trxml_samples = list(load_data_set(self.config, self.trxml_dir))
        self.assertEqual(len(trxml_samples), 10)
        self.assertEqual(list(trxml_samples[0].keys()),
                         ['sec_vacancy.0.sec_vacancy',
                          'derived_org_name.0.derived_org_name',
                          'derived_vac_intermediary.0.derived_vac_intermediary'])
        trxml_samples = list(load_data_set(self.config, self.trxml_dir, train_mode=False))
        self.assertEqual(len(trxml_samples), 10)
        self.assertEqual(list(trxml_samples[0].keys()),
                         ['sec_vacancy.0.sec_vacancy',
                          'derived_org_name.0.derived_org_name',
                          'derived_vac_intermediary.0.derived_vac_intermediary',
                          'Document.0.correlationid',
                          'derived_source_site.0.derived_source_site',
                          'derived_norm_url.0.derived_norm_url'])


    def test_split_csv_data_set(self):
        split_data_set(self.csv_file, ratio=0.8, des=self.test_dir, rand_seed=111)
        train_file = os.path.join(self.test_dir, 'train.csv')
        eval_file = os.path.join(self.test_dir, 'eval.csv')
        train_data = list(load_data_set(self.config, train_file))
        eval_data = list(load_data_set(self.config, eval_file))
        self.assertEqual(len(train_data), 25)
        self.assertEqual(len(eval_data), 5)


    def test_split_trxml_data_set(self):
        split_data_set(self.trxml_dir, ratio=0.8, des=self.test_dir, rand_seed=111)
        train_dir = os.path.join(self.test_dir, 'train')
        eval_dir = os.path.join(self.test_dir, 'eval')
        train_data = list(load_data_set(self.config, train_dir))
        eval_data = list(load_data_set(self.config, eval_dir))
        self.assertEqual(len(train_data), 8)
        self.assertEqual(len(eval_data), 2)
