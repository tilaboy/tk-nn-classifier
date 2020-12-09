', '', '"""unit tests for classifier utils functions"""
import os
import csv
from unittest import TestCase
import tempfile
import shutil

from tk_nn_classifier.exceptions import FileTypeError
from tk_nn_classifier.config import load_config_from_dikt
from tk_nn_classifier.data_loader.csv_loader import CSVLoader, split_csv_file

class CSVLoaderTestCases(TestCase):
    """unit tests"""

    @classmethod
    def setUpClass(self):
        self.csv_file = 'tests/resource/sample.csv'
        self.test_dir = tempfile.mkdtemp()

        config_dikt= {
            "max_lines": 5,
            "model_path": self.test_dir,
            "csv_fields": {
                "features": ["full_text", 'advertiser_name'],
                "class": "source_type",
                "doc_id": "posting_id",
                "extra": ["advertiser_name", "source_website", "source_url"]
            },
            "datasets": {}
        }
        self.config = load_config_from_dikt(config_dikt)
        self.csv_loader = CSVLoader(self.config['csv_fields'])

    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_csv_loader_delimit_type(self):
        self.assertEqual(CSVLoader._delimit_type('ab.csv'), ',')
        self.assertEqual(CSVLoader._delimit_type('ab.tsv'), '\t')
        with self.assertRaisesRegex(FileTypeError, 'file type .* not supported') as cm:
            CSVLoader._delimit_type('ab.usv')

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
        train_examples = list(self.csv_loader.load_train_data(self.csv_file))
        self.assertEqual(len(train_examples), 30)
        self.assertEqual(list(train_examples[0].keys()),
                         ['full_text','advertiser_name', 'source_type'])
        self.assertEqual(
                [row['source_type'] for row in train_examples[:10]],
                ['yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes']
        )
        self.assertEqual(
                [row['advertiser_name'] for row in train_examples[0:10]],
                ['Weather by Healthcare', 'Aquent', 'Mindlance',
                'Elwood Staffing Services, Inc.', 'Accounting Principals',
                'Crystal L. Dunson and Associates, Incorporated',
                'Joseph Michaels, Inc', 'Joseph Michaels, Inc',
                'Crystal L. Dunson and Associates, Incorporated',
                'Accounting Principals']
        )
        self.assertEqual(train_examples[4]['full_text'][:64], self._get_expected_csv_full_text())

    def test_csv_details(self):
        train_examples = list(self.csv_loader.load_detail_data(self.csv_file))
        self.assertEqual(len(train_examples), 30)
        self.assertEqual(list(train_examples[0].keys()),
                         ['full_text','advertiser_name', 'source_type',
                          'posting_id', 'source_website', 'source_url'])
        self.assertEqual(train_examples[4]['full_text'][:64], self._get_expected_csv_full_text())

        self.assertEqual(
                [row['source_type'] for row in train_examples[:10]],
                ['yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes']
        )
        self.assertEqual(
                [row['advertiser_name'] for row in train_examples[0:10]],
                ['Weather by Healthcare', 'Aquent', 'Mindlance',
                'Elwood Staffing Services, Inc.', 'Accounting Principals',
                'Crystal L. Dunson and Associates, Incorporated',
                'Joseph Michaels, Inc', 'Joseph Michaels, Inc',
                'Crystal L. Dunson and Associates, Incorporated',
                'Accounting Principals']
        )
        self.assertEqual(
                [row['posting_id'] for row in train_examples[0:2]],
                ['eb6de587f296446883bdf687705dc239',
                'cc84cf72232245adb8215c287a51b004']
        )
        self.assertEqual(
                [row['source_website'] for row in train_examples[0:2]],
                ['weatherbyhealthcare.com', 'aquent.com']
        )
        self.assertEqual(
                [row['source_url'] for row in train_examples[0:2]],
                ['https://weatherbyhealthcare.com/job/JOB-2599560',
                 'http://aquent.com/find-work/151393']
        )


    def test_split_csv_file(self):
        split_csv_file(self.csv_file, ratio=0.8, des=self.test_dir)
        train_file = os.path.join(self.test_dir, 'train.csv')
        eval_file = os.path.join(self.test_dir, 'eval.csv')
        fieldnames_expected = ['id', 'advertiser_name', 'advertiser_type', 'date',
                           'full_text', 'posting_id', 'source_type',
                           'source_url', 'source_website', 'spider_source']

        with open(train_file, 'r', newline='') as train_fh:
            train_reader = csv.DictReader(train_fh)
            train_fieldnames = train_reader.fieldnames
            self.assertEqual(train_fieldnames, fieldnames_expected)
            self.assertEqual(len(list(train_reader)), 25)

        with open(eval_file, 'r', newline='') as eval_fh:
            eval_reader = csv.DictReader(eval_fh)
            eval_fieldnames = train_reader.fieldnames
            self.assertEqual(eval_fieldnames, fieldnames_expected)
            self.assertEqual(len(list(eval_reader)), 5)

    def _get_expected_csv_full_text(self):
        return '''  Payroll Specialist

  New
  * Location
  Brentwood, California'''
