"""unit tests for classifier utils functions"""
from unittest import TestCase
from tk_nn_classifier.data_loader.common_loader import CommonLoader

class CommonLoaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.config= {
            "max_lines": 5,
            "datasets": {}
        }

    def test_prepare_input_text(self):
        common_loader = CommonLoader(self.config)

        input_text = 'a\nb\nc\nd\ne\nf\ng'
        self.assertEqual(common_loader._prepare_input_text(input_text, True),
                         'a\nb\nc\nd\ne'
                        )
        input_text = 'a\nb\nc\nd\ne\nf\ng'
        self.assertEqual(common_loader._prepare_input_text(input_text),
                         input_text
                        )

    def test_flatten_array(self):
        common_loader = CommonLoader(self.config)
        mixed_array = [[0,1,2],3,4,5,[6,[7,8]],9]
        self.assertEqual(
                list(common_loader._iter_flatten(mixed_array)),
                [0,1,2,3,4,5,6,7,8,9]
        )
