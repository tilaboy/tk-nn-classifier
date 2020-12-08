"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
import tempfile
import shutil
from tk_nn_classifier.data_loader.tf_data_reader import TFDataReader

class TFDataReaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.trxml_dir = 'tests/resource/samples'
        self.csv_file = 'tests/resource/sample.csv'
        self.test_dir = tempfile.mkdtemp()
        self.trxml_label = [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]

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


    def test_tf_data_reader(self):
        self.tf_data_reader = TFDataReader(self.config)

        trxml_examples, trxml_label = self.tf_data_reader.get_data(self.trxml_dir)
        self.assertEqual(len(trxml_examples), 10)

        self.assertEqual(trxml_examples[4][0], self._get_expected_trxml_full_text())

        self.assertEqual(
                trxml_label,
                self.trxml_label
        )

    def _get_expected_trxml_full_text(self):
        return '''TA`s Req for a all-boys Faith school in NW London ASAP on LT

   Recruiter
          ASQ EDUCATION
'''
