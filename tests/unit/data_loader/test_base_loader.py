"""unit tests for classifier utils functions"""
from unittest import TestCase
from tk_nn_classifier.exceptions import ConfigError
from tk_nn_classifier.data_loader.base_loader import BaseLoader
from tk_nn_classifier.config import load_config_from_dikt

class BaseLoaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        config= {
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
                "extra": ["organization_name", "source_type", "source_website"]
            },
            "datasets": {
                "all_data": "foo"
            }
        }
        self.config = load_config_from_dikt(config)

    def test_00_trxml_train_fields(self):
        base_loader = BaseLoader(self.config['trxml_fields'])
        all_fields = ['sec_vacancy.0.sec_vacancy',
                      'derived_vac_intermediary.0.derived_vac_intermediary',
                      'Document.0.correlationid',
                      'derived_org_name.0.derived_org_name',
                      'derived_norm_url.0.derived_norm_url']

        self.assertEqual(
            base_loader._train_fields(),
            all_fields[0:2]
        )

        self.assertEqual(
            base_loader._detail_fields(),
            all_fields
        )

    def test_01_csv_train_fields(self):
        base_loader = BaseLoader(self.config['csv_fields'])
        all_fields = ['full_text', 'advertiser_type', 'posting_id',
                      'organization_name', 'source_type', 'source_website']
        self.assertEqual(
            base_loader._train_fields(),
            all_fields[0:2]
        )

        self.assertEqual(
            base_loader._detail_fields(),
            all_fields
        )

    def test_02_missing_fields(self):
        del self.config['csv_fields']['features']
        with self.assertRaisesRegex(ConfigError, 'Missing/Wrong') as cm:
            base_loader = BaseLoader(self.config['csv_fields'])

        del self.config['trxml_fields']['class']
        with self.assertRaisesRegex(ConfigError, 'Missing/Wrong') as cm:
            base_loader = BaseLoader(self.config['trxml_fields'])
