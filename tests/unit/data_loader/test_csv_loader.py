', '', '"""unit tests for classifier utils functions"""
import os
import csv
from unittest import TestCase
import tempfile
import shutil
from tk_nn_classifier.data_loader.csv_loader import CSVLoader

class CSVLoaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
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
        self.csv_loader = CSVLoader(self.config)

    def tearDown(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_csv_train_fields_list (self):
        self.assertEqual(self.csv_loader._train_fields(),
                [['full_text','advertiser_name'],
                 'source_type']
        )

    def test_csv_detailed_fields_list (self):
        self.assertEqual(self.csv_loader._detail_fields(),
                [['full_text', 'advertiser_name'],
                 'source_type',
                 'posting_id',
                 'advertiser_name',
                 'source_website',
                 'source_url']
        )

    def test_csv_reading(self):
        train_examples = list(self.csv_loader.get_train_data(self.csv_file))
        self.assertEqual(len(train_examples), 30)
        full_text, categories = zip(*train_examples)
        self.assertEqual(categories[4], 'yes')
        self.assertEqual(categories[5], 'no')

        self.assertEqual(full_text[4][0], self._get_expected_csv_full_text())

    def test_csv_details(self):
        train_examples = list(self.csv_loader.get_details(self.csv_file))
        self.assertEqual(len(train_examples), 30)
        text, categories, doc_ids, org_names, sites, urls = zip(*train_examples)
        self.assertEqual(text[4][0], self._get_expected_csv_full_text())

        self.assertEqual(
                categories[:10],
                ('yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes')
        )
        self.assertEqual(
                doc_ids[0:2],
                ('eb6de587f296446883bdf687705dc239',
                'cc84cf72232245adb8215c287a51b004')
        )
        self.assertEqual(
                org_names[:10],
                ('Weather by Healthcare', 'Aquent', 'Mindlance',
                'Elwood Staffing Services, Inc.', 'Accounting Principals',
                'Crystal L. Dunson and Associates, Incorporated',
                'Joseph Michaels, Inc', 'Joseph Michaels, Inc',
                'Crystal L. Dunson and Associates, Incorporated',
                'Accounting Principals')
        )
        self.assertEqual(
                sites[:2],
                ('weatherbyhealthcare.com', 'aquent.com')
        )
        self.assertEqual(
                urls[0:2],
                ('https://weatherbyhealthcare.com/job/JOB-2599560', 'http://aquent.com/find-work/151393')
        )


    def test_split_docs_on_ratio(self):
        header, train_rows, eval_rows = CSVLoader._split_docs_on_ratio(self.csv_file, ratio=0.8)
        header_expected = ['id', 'advertiser_name', 'advertiser_type', 'date',
                           'full_text', 'posting_id', 'source_type',
                           'source_url', 'source_website', 'spider_source']

        self.assertEqual(header, header_expected)
        self.assertEqual(len(train_rows), 24)
        self.assertEqual(len(eval_rows), 6)

    def test_split_data(self):
        self.csv_loader.split_data(self.csv_file, ratio=0.8, des=self.test_dir)
        train_file = os.path.join(self.test_dir, 'train.csv')
        eval_file = os.path.join(self.test_dir, 'eval.csv')
        header_expected = ['id', 'advertiser_name', 'advertiser_type', 'date',
                           'full_text', 'posting_id', 'source_type',
                           'source_url', 'source_website', 'spider_source']

        with open(train_file, 'r', newline='') as train_fh:
            rows = list(csv.reader(train_fh))
            header = rows.pop(0)
        self.assertEqual(header, header_expected)
        self.assertEqual(len(rows), 24)

        with open(eval_file, 'r', newline='') as eval_fh:
            rows = list(csv.reader(eval_fh))
            header = rows.pop(0)
        self.assertEqual(header, header_expected)
        self.assertEqual(len(rows), 6)


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
