"""unit tests for classifier utils functions"""
import os
import tempfile
import shutil
from unittest import TestCase
from recruitment_agency_detector.data_loader.data_reader import DataReader
from recruitment_agency_detector.data_loader.data_reader import _iter_flatten

class DataReaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.trxml_dir = 'tests/resource/samples'
        self.csv_file = 'tests/resource/us_small.csv'
        self.config= {
            "max_lines": 50,
            "model_path": 'foo',
            "trxml_fields": {
                "features": ["sec_vacancy.0.sec_vacancy"],
                "class": "derived_vac_intermediary.0.derived_vac_intermediary",
                "doc_id": "Document.0.correlationid",
                "extra": ["derived_org_name.0.derived_org_name",
                    "derived_source_site.0.derived_source_site",
                    "derived_norm_url.0.derived_norm_url"]
            },
            "csv_fields": {
                "features": ["full_text"],
                "class": "source_type",
                "doc_id": "posting_id",
                "extra": ["advertiser_name", "source_website", "source_url"]
            },
            "datasets": {}
        }
        self.data_reader = DataReader(self.config)

    def test_flatten_array(self):
        mixed_array = [[0,1,2],3,4,5,[6,[7,8]],9]
        self.assertEqual(list(_iter_flatten(mixed_array)), [0,1,2,3,4,5,6,7,8,9])



    def test_trxml_train_fields_list (self):
        self.assertEqual(self.data_reader._train_fields(self.trxml_dir),
                [['sec_vacancy.0.sec_vacancy'],
                 'derived_vac_intermediary.0.derived_vac_intermediary']
        )

    def test_trxml_detailed_fields_list (self):
        self.assertEqual(self.data_reader._detail_fields(self.trxml_dir),
                [['sec_vacancy.0.sec_vacancy'],
                 'derived_vac_intermediary.0.derived_vac_intermediary',
                 'Document.0.correlationid',
                 'derived_org_name.0.derived_org_name',
                 'derived_source_site.0.derived_source_site',
                 'derived_norm_url.0.derived_norm_url']
        )

    def test_csv_train_fields_list (self):
        self.assertEqual(self.data_reader._train_fields(self.csv_file),
                [['full_text'],
                 'source_type']
        )

    def test_csv_detailed_fields_list (self):
        self.assertEqual(self.data_reader._detail_fields(self.csv_file),
                [['full_text'],
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
                'Samuel Frank', None)
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


    def _get_expected_csv_full_text(self):
        return '''  Payroll Specialist

  New
  * Location
  Brentwood, California
  * Category
  Payroll / Benefits
  * Job reference:
  US_EN_5_849088_2696415
  * Job type
  Contract/Temporary

  We are looking for an experienced Payroll professional for a temporary to hire position (3 months) in the Brentwood area. This company is working with the county of Contra Costa and is growing rapidly. This position is open based on the growth of the company and the increase of work and need for a new position. They are looking to pay this person up to 75K based on experience.

  The Position Will Be Responsible For:
  * Reconciling benefits (medical, dental, vision)
  * Reconciling retirement
  * Processing Payroll twice a month
  * Correcting any discrepancies in Payroll

  The Requirements Are:
  * High attention to detail
  * Someone who is willing to dig deep into issues that arise
  * Union experience is a must
  * At least 5 years of experience
  * Degree is helpful but mandatory

  If you are interested in this or other Payroll positions available through Accounting Principals please submit your resume today at www.accountingprincipals.com !

  Please apply with your CV to:

  More Information

  * Save for later
  Job saved

  * Apply with us'''


    def _get_expected_trxml_full_text(self):
        return '''TA`s Req for a all-boys Faith school in NW London ASAP on LT

   Recruiter
          ASQ EDUCATION

   Location
          UK

   Salary
          Competitive

   Posted
          10 Jul 2019

   Closes
          09 Aug 2019

   Ref
          1192248017

   Sector
          Education

   Contract Type
          Permanent

   Hours
          Full Time

   - Teaching Assistants Needed - All boys Secondary Faith School based within the SEN Department - North West London - ASAP Start

   We are currently looking for Experienced Teaching Assistants, HLTA or Newly Qualified Teacher who are open to support work to work within All-boys secondary Faith School located in North West London. This is for an ASAP start and is on a full time temp- perm basis. We are ideally looking for proactive and enthusiastic candidates with a real drive to make a difference.

   The role

   If you have a love for working with young students and truly desire to make a difference to their learning experience, supporting them to achieve their goals and ultimate desires, enabling them to move on to a prosperous future, this role is perfect for you. You must be an experienced Teaching Assistant with strong literacy and numeracy knowledge.

   To be considered a real asset to the school, you must demonstrate:
     * Experience of working with Special Educational Needs or open to gaining experience in SEN
     * Ideally HLTA or very experienced
     * Confidently able to provide 1:1 and small group support work
     * A genuine passion for education and supporting the development of others
     * Strong interpersonal skills
     * Great organisational skills with a friendly and professional manor

   The school

   The School is an exciting place to work and learn as they take seriously the fact that their children and young people only get one chance at an excellent education. To this end they strive to ensure that everyone in the School community is able to take advantage of the best available facilities and opportunities during their time with them. They aim to equip their children and young people with the skills necessary for life in an ever-changing, highly technological world whether it be their intention to progress through their sixth form and on to university or into employment.

   To apply for this position, please send your CV ASAP to or give him a call on'''
