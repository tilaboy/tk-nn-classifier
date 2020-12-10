"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
import tempfile
import shutil
from tk_nn_classifier.data_loader.trxml_loader import TRXMLLoader, split_trxml_set

class TRXMLLoaderTestCases(TestCase):
    """unit tests for trxml loader class"""

    @classmethod
    def setUpClass(self):
        self.trxml_dir = 'tests/resource/samples'
        self.test_dir = tempfile.mkdtemp()

        config= {
            "trxml_fields": {
                "features": ["sec_vacancy.0.sec_vacancy", 'derived_org_name.0.derived_org_name'],
                "class": "derived_vac_intermediary.0.derived_vac_intermediary",
                "doc_id": "Document.0.correlationid",
                "extra": ["derived_org_name.0.derived_org_name",
                    "derived_source_site.0.derived_source_site",
                    "derived_norm_url.0.derived_norm_url"]
            }
        }
        self.trxml_loader = TRXMLLoader(config['trxml_fields'])

    @classmethod
    def tearDownClass(self):
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

    def _unpack_field(self, examples, field):
        if '.' not in field:
            field = field + '.0.' + field
        return [trxml[field] for trxml in examples]

    def test_trxml_reading(self):
        examples = list(self.trxml_loader.load_train_data(self.trxml_dir))
        self.assertEqual(len(examples), 10)
        self.assertEqual(
            self._unpack_field(examples, 'derived_vac_intermediary'),
            ['yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes']
        )
        self.assertEqual(examples[4]['sec_vacancy.0.sec_vacancy'][:100],
                         self._get_expected_trxml_full_text())
        self.assertEqual(
            self._unpack_field(examples, 'derived_org_name.0.derived_org_name'),
            ['Blusource', 'Red Eagle', 'NLB Solutions', 'Estio Healthcare Ltd',
             'ASQ EDUCATION', 'Class People', 'West Suffolk Council',
             'Estio Healthcare Ltd', 'Samuel Frank', '']
        )


    def test_trxml_details(self):
        examples = list(self.trxml_loader.load_detail_data(self.trxml_dir))
        self.assertEqual(len(examples), 10)
        self.assertEqual(examples[4]['sec_vacancy.0.sec_vacancy'][:100],
                         self._get_expected_trxml_full_text())
        self.assertEqual(
            self._unpack_field(examples, 'derived_vac_intermediary'),
            ['yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes']
        )
        self.assertEqual(
            self._unpack_field(examples, 'Document.0.correlationid')[:2],
            ['3cdb9e9d9e7848909c7bd1ecd2abd99f',
             '46cabf28675c4ec09cacac68b3a5180a']
        )
        self.assertEqual(
            self._unpack_field(examples, 'derived_org_name'),
            ['Blusource', 'Red Eagle', 'NLB Solutions',
             'Estio Healthcare Ltd', 'ASQ EDUCATION', 'Class People',
             'West Suffolk Council', 'Estio Healthcare Ltd',
             'Samuel Frank', '']
        )
        self.assertEqual(
            self._unpack_field(examples, 'derived_source_site')[:2],
            ['reed.co.uk', 'cv-library.co.uk']
        )
        self.assertEqual(
            self._unpack_field(examples, 'derived_norm_url')[:2],
            ['reed.co.uk/jobs/finance-director/37941677',
             'cv-library.co.uk/job/210057084/audit-senior-academies-expert']
        )


    def test_split_set(self):
        split_trxml_set(self.trxml_dir, ratio=0.8, des=self.test_dir)
        train_dir = os.path.join(self.test_dir, 'train')
        eval_dir = os.path.join(self.test_dir, 'eval')

        train_files = os.listdir(train_dir)
        self.assertEqual(len(train_files), 8)
        eval_files = os.listdir(eval_dir)
        self.assertEqual(len(eval_files), 2)

    def _get_expected_trxml_full_text(self):
        return '''TA`s Req for a all-boys Faith school in NW London ASAP on LT

   Recruiter
          ASQ EDUCATION

'''
