"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
import tempfile
import shutil
from tk_nn_classifier.data_loader.trxml_loader import TRXMLLoader

class TRXMLLoaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.trxml_dir = 'tests/resource/samples'
        self.test_dir = tempfile.mkdtemp()
        self.trxml_label = ('yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes')

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
        self.trxml_loader = TRXMLLoader(self.config)

    def tearDown(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_trxml_train_fields_list (self):
        self.assertEqual(self.trxml_loader._train_fields(),
                [['sec_vacancy.0.sec_vacancy', 'derived_org_name.0.derived_org_name'],
                 'derived_vac_intermediary.0.derived_vac_intermediary']
        )

    def test_trxml_detailed_fields_list (self):
        self.assertEqual(self.trxml_loader._detail_fields(),
                [['sec_vacancy.0.sec_vacancy', 'derived_org_name.0.derived_org_name'],
                 'derived_vac_intermediary.0.derived_vac_intermediary',
                 'Document.0.correlationid',
                 'derived_org_name.0.derived_org_name',
                 'derived_source_site.0.derived_source_site',
                 'derived_norm_url.0.derived_norm_url']
        )

    def test_trxml_reading(self):
        train_examples = list(self.trxml_loader.get_train_data(self.trxml_dir))
        self.assertEqual(len(train_examples), 10)
        full_text, categories = zip(*train_examples)
        self.assertEqual(
                categories,
                self.trxml_label
        )
        self.assertEqual(full_text[4][0], self._get_expected_trxml_full_text())


    def test_trxml_details(self):
        train_examples = list(self.trxml_loader.get_details(self.trxml_dir))
        self.assertEqual(len(train_examples), 10)
        text, categories, doc_ids, org_names, sites, urls = zip(*train_examples)
        self.assertEqual(text[4][0], self._get_expected_trxml_full_text())
        self.assertEqual(
                categories,
                self.trxml_label
        )
        self.assertEqual(
                doc_ids[0:2],
                ('3cdb9e9d9e7848909c7bd1ecd2abd99f',
                '46cabf28675c4ec09cacac68b3a5180a')
        )
        self.assertEqual(
                org_names,
                ('Blusource', 'Red Eagle', 'NLB Solutions',
                'Estio Healthcare Ltd', 'ASQ EDUCATION', 'Class People',
                'West Suffolk Council', 'Estio Healthcare Ltd',
                'Samuel Frank', '')
        )
        self.assertEqual(
                sites[0:2],
                ('reed.co.uk', 'cv-library.co.uk')
        )
        self.assertEqual(
                urls[0:2],
                ('reed.co.uk/jobs/finance-director/37941677', 'cv-library.co.uk/job/210057084/audit-senior-academies-expert')
        )


    def test_split_data_trxml(self):
        train_files, eval_files = TRXMLLoader._split_docs_on_ratio(self.trxml_dir, ratio=0.8)
        self.assertEqual(len(train_files), 8)
        self.assertEqual(len(eval_files), 2)

    def test_split_data(self):
        self.trxml_loader.split_data(self.trxml_dir, ratio=0.8, des=self.test_dir)
        train_dir = os.path.join(self.test_dir, 'train')
        eval_dir = os.path.join(self.test_dir, 'eval')

        train_files = os.listdir(train_dir)
        self.assertEqual(len(train_files), 8)

    def _get_expected_csv_full_text(self):
        return '''  Payroll Specialist

  New
  * Location
  Brentwood, California'''


    def _get_expected_trxml_full_text(self):
        return '''TA`s Req for a all-boys Faith school in NW London ASAP on LT

   Recruiter
          ASQ EDUCATION
'''
