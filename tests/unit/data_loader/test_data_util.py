from unittest import TestCase
from tk_nn_classifier.data_loader.data_utils import iter_flatten, file_ext

class DataUtilsTestCases(TestCase):
    def test_iter_flatten(self):
        mixed_array = [[[0,1,2],3],4,5,[6,[7,8]],9]
        self.assertEqual(
                list(iter_flatten(mixed_array)),
                [0,1,2,3,4,5,6,7,8,9]
        )

    def test_file_type(self):
        cases = [
            {'file_path': 'foo.new', 'ext': 'new'},
            {'file_path': 'a/b/foo.csv', 'ext': 'csv'},
            {'file_path': 'a/b/bar.c', 'ext': 'c'},
            {'file_path': 'noext', 'ext': ''}
        ]
        for case in cases:
            self.assertEqual(
                file_ext(case['file_path']),
                case['ext'],
                msg=f"{case['file_path']} => {case['ext']}"
            )
