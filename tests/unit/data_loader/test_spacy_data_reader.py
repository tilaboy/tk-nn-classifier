"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
import tempfile
import shutil
from tk_nn_classifier.data_loader.spacy_data_reader import SpacyDataReader

class SpacyDataReaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.trxml_dir = 'tests/resource/samples'
        self.csv_file = 'tests/resource/sample.csv'
        self.test_dir = tempfile.mkdtemp()

        self.config= {
            "max_lines": 5,
            "model_path": self.test_dir,
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
            "datasets": {}
        }

    def tearDown(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_spacy_data_reader(self):
        self.sp_data_reader = SpacyDataReader(self.config)

        data_set = self.sp_data_reader.get_data(self.trxml_dir)
        self.assertEqual(len(data_set), 10)

        full_text, categories = zip(*data_set)
        self.assertEqual(categories[4], {'no': True, 'yes': False})
        self.assertEqual(categories[5], {'no': True, 'yes': False})
        expected_org_name = 'ASQ EDUCATION'
        self.assertEqual(
                full_text[4],
                self._get_expected_trxml_full_text() + '\n' + expected_org_name
        )

    def _get_expected_trxml_full_text(self):
        return '''TA`s Req for a all-boys Faith school in NW London ASAP on LT

   Recruiter
          ASQ EDUCATION
'''
