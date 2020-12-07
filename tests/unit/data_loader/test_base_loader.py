"""unit tests for classifier utils functions"""
from unittest import TestCase
from tk_nn_classifier.data_loader.base_loader import BaseLoader

class BaseLoaderTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.config= {
            "max_lines": 5,
            "datasets": {}
        }

    def test_prepare_input_text(self):
        base_loader = BaseLoader(self.config)

        input_text = 'a\nb\nc\nd\ne\nf\ng'
        self.assertEqual(base_loader._prepare_input_text(input_text, True),
                         'a\nb\nc\nd\ne'
                        )
        input_text = 'a\nb\nc\nd\ne\nf\ng'
        self.assertEqual(base_loader._prepare_input_text(input_text),
                         input_text
                        )

    def test_flatten_array(self):
        base_loader = BaseLoader(self.config)
        mixed_array = [[0,1,2],3,4,5,[6,[7,8]],9]
        self.assertEqual(
                list(base_loader._iter_flatten(mixed_array)),
                [0,1,2,3,4,5,6,7,8,9]
        )
