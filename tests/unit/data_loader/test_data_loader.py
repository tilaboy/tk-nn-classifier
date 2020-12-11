from unittest import TestCase

from tk_nn_classifier.data_loader import load_data_set, _select_data_reader
from tk_nn_classifier.data_loader.csv_loader import CSVLoader
from tk_nn_classifier.data_loader.trxml_loader import TRXMLLoader
from tk_nn_classifier.exceptions import FileTypeError

class DataLoaderTestCase(TestCase):
    def setUp(self):
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

    def test_select_data_reader(self):
        data_reader = _select_data_reader(self.config['csv_fields'],
                                          'tests/resource/sample.csv')

        self.assertIsInstance(data_reader, CSVLoader)

        data_reader = _select_data_reader(self.config['trxml_fields'],
                                          'tests/resource/sample_trxmls')
        self.assertIsInstance(data_reader, TRXMLLoader)

        with self.assertRaisesRegex(FileNotFoundError, 'not found') as cm:
            data_reader = _select_data_reader(self.config['csv_fields'],
                                              'tests/resource/no_exist.csv')

        with self.assertRaisesRegex(FileNotFoundError, 'not found') as cm:
            data_reader = _select_data_reader(self.config['trxml_fields'],
                                              'tests/resource/no_exist')

        with self.assertRaisesRegex(FileTypeError, 'not support') as cm:
            data_reader = _select_data_reader(self.config['trxml_fields'],
                                              'tests/resource/label_mapper.json')

    def test_load_data_set(self):
        pass
