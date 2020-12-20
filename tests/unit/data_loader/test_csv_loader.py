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
        self.csv_file = 'tests/resource/eval.csv'
        self.test_dir = tempfile.mkdtemp()

        config= {
            "csv_fields": {
                "features": ["full_text", 'organization_name'],
                "class": "advertiser_type",
                "doc_id": "posting_id",
                "extra": ["organization_name", "source_url"]
            }
        }
        self.csv_loader = CSVLoader(config['csv_fields'])

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
                [['full_text','organization_name'],
                 'advertiser_type']
        )

    def test_csv_detailed_fields_list (self):
        self.assertEqual(self.csv_loader._detail_fields(),
                [['full_text', 'organization_name'],
                 'advertiser_type',
                 'posting_id',
                 ['organization_name',
                 'source_url']]
        )

    def test_csv_reading(self):
        train_examples = list(self.csv_loader.load_train_data(self.csv_file))
        self.assertEqual(len(train_examples), 64)
        self.assertEqual(list(train_examples[0].keys()),
                         ['full_text','organization_name', 'advertiser_type'])

        self.assertEqual(
                [row['advertiser_type'] for row in train_examples[:10]],
                self._expected_class_first_10()
        )
        self.assertEqual(
                [row['organization_name'] for row in train_examples[0:10]],
                self._expected_orgname_first_10()
        )
        self.assertEqual(train_examples[4]['full_text'][:100],
                         self._expected_csv_text())

    def test_csv_details(self):
        train_examples = list(self.csv_loader.load_detail_data(self.csv_file))
        self.assertEqual(len(train_examples), 64)
        self.assertEqual(list(train_examples[0].keys()),
                         ['full_text','organization_name', 'advertiser_type',
                          'posting_id', 'source_url'])
        self.assertEqual(train_examples[4]['full_text'][:100],
                         self._expected_csv_text())

        self.assertEqual(
                [row['advertiser_type'] for row in train_examples[:10]],
                self._expected_class_first_10()
        )
        self.assertEqual(
                [row['organization_name'] for row in train_examples[0:10]],
                self._expected_orgname_first_10()
        )
        self.assertEqual(
                [row['posting_id'] for row in train_examples[0:10]],
                self._expected_docid_first_10()
        )
        print([row['source_url'] for row in train_examples[0:10]])
        self.assertEqual(
                [row['source_url'] for row in train_examples[0:10]],
                self._expected_url_first_10()
        )


    def test_split_csv_file(self):
        split_csv_file(self.csv_file, ratio=0.8, des=self.test_dir)
        train_file = os.path.join(self.test_dir, 'train.csv')
        eval_file = os.path.join(self.test_dir, 'eval.csv')
        fieldnames_expected = ['id', 'posting_id', 'country',
                               'advertiser_type', 'organization_name',
                               'source_url', 'full_text']

        with open(train_file, 'r', newline='') as train_fh:
            train_reader = csv.DictReader(train_fh)
            train_fieldnames = train_reader.fieldnames
            print(train_fieldnames)
            self.assertEqual(train_fieldnames, fieldnames_expected)
            self.assertEqual(len(list(train_reader)), 50)

        with open(eval_file, 'r', newline='') as eval_fh:
            eval_reader = csv.DictReader(eval_fh)
            eval_fieldnames = train_reader.fieldnames
            self.assertEqual(eval_fieldnames, fieldnames_expected)
            self.assertEqual(len(list(eval_reader)), 14)

    def _expected_csv_text(self):
        return '''Assistant Store Manager, adidas Factory Outlet, O2 Greenwich, London

   London | United Kingdom | R'''


    def _expected_class_first_10(self):
        return ['yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no']

    def _expected_orgname_first_10(self):
        return ['JMF Associates', 'Applause IT',
                'Prestige Recruitment Services', 'Berry Recruitment',
                'adidas Group', 'Encore Personnel', 'Boots Company PLC',
                'Elite Personnel', 'Advantage Resourcing', 'Computacenter PLC']

    def _expected_docid_first_10(self):
        return ['7b6defe63d32495b88ff0ecfcbfe4f9a',
                '7dc15e05679c4165a32c59e8af37e542',
                '8d81f4293d4f41ef8d49e845750915ec',
                'd8a068665ccc4a07a1fb941f5b6c1d4a',
                '98250d7bbc9d46e79bfaeb124692e120',
                'c958b1b08fbb4763aea853f3ca9929f1',
                'f10eb6361751493ead96cd7186469e55',
                '6aa5f7ec1c864838ad9a2dadf6c95001',
                '80892cec70214579a8a901e8bc5fdc9f',
                '677e430b55d9480f93c9e95b0969a0e8']

    def _expected_url_first_10(self):
        return [
            'https://www.jmfassociates.co.uk/job-search/1129-finance-manager/accountancy-commerce-industry/south-east-london/job',
            'https://www.reed.co.uk/jobs/dynamics-application-support-analyst/38296089',
            'https://jobs.prestigerecruitmentgroup.com/jobs/account-manager/',
            'https://www.reed.co.uk/jobs/home-manager/38350988',
            'https://careers.adidas-group.com/jobs/assistant-store-manager-adidas-factory-outlet-o2-gree-192410?locale=en',
            'https://www.encorepersonnel.co.uk/job/aerospace-paint-sprayer/',
            'https://www.boots.jobs/jobs/103951br-trainee-pharmacy-advisor-trainee-dispenser-2/',
            'https://independentjobs.independent.co.uk/job/13997116/sales-administrator/',
            'https://www.advantageresourcing.co.uk/candidates/search-opportunities?p_p_id=eRecruit_WAR_eRecruitportlet&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-2&p_p_col_pos=1&p_p_col_count=2&_eRecruit_WAR_eRecruitportlet_positionId=826769&_eRecruit_WAR_eRecruitportlet_jspPage=%2FdisplayPositionuk.jsp',
            'https://jobsearch.computacenter.com/jobs/job/Network-Security-Consultant-Symantec/2920']
