"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
import tempfile
import shutil
from tk_nn_classifier.data_loader import DataReader
from tk_nn_classifier.data_loader import TFDataReader
from tk_nn_classifier.data_loader import SpacyDataReader
from tk_nn_classifier.data_loader import CommonLoader
from tk_nn_classifier.data_loader import TRXMLLoader
from tk_nn_classifier.data_loader import CSVLoader

class DataReaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.trxml_dir = 'tests/resource/samples'
        self.csv_file = 'tests/resource/us_small.csv'
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
        self.data_reader = DataReader(self.config)

    def tearDown(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_flatten_array(self):
        common_reader = CommonLoader(self.config)
        mixed_array = [[0,1,2],3,4,5,[6,[7,8]],9]
        self.assertEqual(
                list(common_reader._iter_flatten(mixed_array)),
                [0,1,2,3,4,5,6,7,8,9]
        )

    def test_trxml_train_fields_list (self):
        self.assertEqual(self.data_reader._train_fields(self.trxml_dir),
                [['sec_vacancy.0.sec_vacancy', 'derived_org_name.0.derived_org_name'],
                 'derived_vac_intermediary.0.derived_vac_intermediary']
        )

    def test_trxml_detailed_fields_list (self):
        self.assertEqual(self.data_reader._detail_fields(self.trxml_dir),
                [['sec_vacancy.0.sec_vacancy', 'derived_org_name.0.derived_org_name'],
                 'derived_vac_intermediary.0.derived_vac_intermediary',
                 'Document.0.correlationid',
                 'derived_org_name.0.derived_org_name',
                 'derived_source_site.0.derived_source_site',
                 'derived_norm_url.0.derived_norm_url']
        )

    def test_csv_train_fields_list (self):
        self.assertEqual(self.data_reader._train_fields(self.csv_file),
                [['full_text','advertiser_name'],
                 'source_type']
        )

    def test_csv_detailed_fields_list (self):
        self.assertEqual(self.data_reader._detail_fields(self.csv_file),
                [['full_text', 'advertiser_name'],
                 'source_type',
                 'posting_id',
                 'advertiser_name',
                 'source_website',
                 'source_url']
        )


    def test_trxml_reading(self):
        train_examples = list(self.data_reader.get_data_set(self.trxml_dir))
        self.assertEqual(len(train_examples), 10)
        full_text, categories = zip(*train_examples)
        self.assertEqual(
                categories,
                ('no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no')
        )
        self.assertEqual(full_text[4][0], self._get_expected_trxml_full_text())


    def test_trxml_details(self):
        train_examples = list(self.data_reader.get_data_set_with_detail(self.trxml_dir))
        self.assertEqual(len(train_examples), 10)
        text, categories, doc_ids, org_names, sites, urls = zip(*train_examples)
        self.assertEqual(text[4][0], self._get_expected_trxml_full_text())
        self.assertEqual(
                categories,
                ('no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no')
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


    def test_csv_reading(self):
        train_examples = list(self.data_reader.get_data_set(self.csv_file))
        self.assertEqual(len(train_examples), 30)
        full_text, categories = zip(*train_examples)
        self.assertEqual(categories[4], 'yes')
        self.assertEqual(categories[5], 'no')

        self.assertEqual(full_text[4][0], self._get_expected_csv_full_text())

    def test_csv_details(self):
        train_examples = list(self.data_reader.get_data_set_with_detail(self.csv_file))
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

    def test_tf_data_reader(self):
        self.tf_data_reader = TFDataReader(self.config)

        trxml_examples, trxml_label = self.tf_data_reader.get_data(self.trxml_dir)
        self.assertEqual(len(trxml_examples), 10)

        self.assertEqual(trxml_examples[4][0], self._get_expected_trxml_full_text())

        self.assertEqual(
                trxml_label,
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        )

    def test_spacy_data_reader(self):
        self.sp_data_reader = SpacyDataReader(self.config)

        data_set = self.sp_data_reader.get_data(self.trxml_dir)
        self.assertEqual(len(data_set), 10)

        full_text, categories = zip(*data_set)
        print(full_text)
        self.assertEqual(categories[4], {'no': True, 'yes': False})
        self.assertEqual(categories[5], {'no': True, 'yes': False})
        expected_org_name = 'ASQ EDUCATION'
        self.assertEqual(
                full_text[4],
                self._get_expected_trxml_full_text() + '\n' + expected_org_name
        )

    def test_split_data_trxml(self):
        train_files, eval_files = CSVLoader.split_data_on_ratio(self.trxml_dir)
        self.assertEqual(len(train_files), 8)
        self.assertEqual(len(eval_files), 2)


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
