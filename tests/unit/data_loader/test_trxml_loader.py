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
        self.trxml_dir = 'tests/resource/sample_trxmls'
        self.test_dir = tempfile.mkdtemp()

        config= {
            "trxml_fields": {
                "features": ["sec_vacancy.0.sec_vacancy", 'derived_org_name.0.derived_org_name'],
                "class": "derived_vac_intermediary.0.derived_vac_intermediary",
                "doc_id": "Document.0.correlationid",
                "extra": ["derived_org_name.0.derived_org_name",
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
                 ['derived_org_name.0.derived_org_name',
                 'derived_norm_url.0.derived_norm_url']]
        )

    def _unpack_field(self, examples, field):
        if '.' not in field:
            field = field + '.0.' + field
        return [trxml[field] for trxml in examples]

    def test_trxml_reading(self):
        examples = list(self.trxml_loader.load_train_data(self.trxml_dir))
        self.assertEqual(len(examples), 64)
        self.assertEqual(
            self._unpack_field(examples, 'derived_vac_intermediary')[:10],
            self._expected_class_first_10()
        )
        self.assertEqual(examples[4]['sec_vacancy.0.sec_vacancy'][:100],
                         self._expected_trxml_text())
        self.assertEqual(
            self._unpack_field(examples, 'derived_org_name.0.derived_org_name')[:10],
            self._expected_orgname_first_10()
        )


    def test_trxml_details(self):
        examples = list(self.trxml_loader.load_detail_data(self.trxml_dir))
        self.assertEqual(len(examples), 64)
        self.assertEqual(examples[4]['sec_vacancy.0.sec_vacancy'][:100],
                         self._expected_trxml_text())
        self.assertEqual(
            self._unpack_field(examples, 'derived_vac_intermediary')[:10],
            self._expected_class_first_10()
        )
        self.assertEqual(
            self._unpack_field(examples, 'Document.0.correlationid')[:10],
            self._expected_docid_first_10()
        )
        self.assertEqual(
            self._unpack_field(examples, 'derived_org_name')[:10],
            self._expected_orgname_first_10()
        )
        self.assertEqual(
            self._unpack_field(examples, 'derived_norm_url')[:10],
            self._expected_url_first_10()
        )


    def test_split_set(self):
        split_trxml_set(self.trxml_dir, ratio=0.8, des=self.test_dir)
        train_dir = os.path.join(self.test_dir, 'train')
        eval_dir = os.path.join(self.test_dir, 'eval')

        train_files = os.listdir(train_dir)
        self.assertEqual(len(train_files), 50)
        eval_files = os.listdir(eval_dir)
        self.assertEqual(len(eval_files), 14)

    def _expected_class_first_10(self):
        return ['no', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'no']

    def _expected_orgname_first_10(self):
        return ['Cumbria County Council', 'Prospero Recruitment Ltd',
                'RIG Healthcare', 'vodafone', "Domino's Pizza",
                'RIG Healthcare', 'Iceland Foods Ltd',
                'Talascend LLC', 'Macildowie Associates',
                'All Bar One O2']

    def _expected_trxml_text(self):
        return '''Delivery Cyclist

   Store:
   Salisbury

   Shift Pattern:
   Days, Evenings, Nights

   Hours:
   '''

    def _expected_docid_first_10(self):
        return ['020f2bef4f9f4a9ca1fabccb0f2b3f9e',
                '03099ae04d8940a6b4f2eb7cb8cc4178',
                '0a1b515e707f4a49808e12d72947d431',
                '0d5bb67e78dc457f9262f8a5139c284c',
                '0e0bc17a2cb04c4eaef26ab2d13bd1d6',
                '1d35810e21744d21b0a28ba45acac6f8',
                '1ef2c57de6224a7b955c13aa7a915941',
                '21cf0cc0fbb04c6cbf6393c23725934a',
                '22583f36f6f440039d2aabb510ede5b7',
                '25c96bf4ab2f482685ebb83f2b53b7ea']

    def _expected_url_first_10(self):
        return [
            'cumbria.gov.uk/trentl_webrecruitment/wrd/run/etrec107gf.open?vacancy_id=004408pzjo&wvid=3697840jjb',
            'prosperoteaching.com/jobs/english-nqt-highgate-north-west-london-september-2019',
            'righealthcare.co.uk/job-details/cardiac-physiology/manchester/cardiac-physiologist/?id=238.052.17881&key=rad',
            'vodafone.com/job/retail-adviser-40hrs-week-macclesfield-in-macclesfield-cheshire-jid-26325',
            'isw.changeworknow.co.uk/dominos/bbbgfprhzlrrayzmudqfaj',
            'righealthcare.co.uk/job-details/category/location/locum-physiotherapist--msk--band-6/?id=8160.247.19723&key=ot',
            'app.kallidusrecruit.com/iceland/vacancyinformation.aspx?vid=38655',
            'talascend.com/99641',
            'macildowie.com/jobs/finance-analyst-global-business-nottingham',
            'harri.com/all-bar-one-o2-2/job/1030195-waiting-staff']
