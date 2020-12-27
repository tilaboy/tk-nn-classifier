"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
import tempfile
import shutil
from tk_nn_classifier.model_input.tokenizer import tokenize

class TokenizerTestCases(TestCase):
    """unit tests"""

    def test_tk_token_and_norm (self):
        text = 'abc http://www.yahoo.com email aa@yahoo.com 2012 1234'
        self.assertEqual(tokenize(text),
                         ['ABC', 'xxURLxx', 'EMAIL', 'xxEMAILxx', 'xxYEARxx', '1111']
                        )
