"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
from tk_nn_classifier.classifiers.utils import TrainHelper

class ClassifierUtilTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.train_helper = TrainHelper()

    def test_max_dict_value(self):
        cats_dict = [
            {'a': 0.5, 'b': 0.3, 'c': 0.2},
            {'a': 0.1, 'b': 0.3, 'c': 0.6},
            {'a': 0, 'b': 0.9, 'c': 0.1}
        ]
        self.assertEqual(TrainHelper.max_dict_value(cats_dict), ['a', 'c', 'b'])



    def test_confusion_binary(self):
        eval = [1,1,1,0,1,0]
        gold = [1,0,1,0,1,0]

        conf_matrix = \
                self.train_helper._evaluate_confusion_matrix(
                        eval, gold)
        self.assertEqual(conf_matrix.iloc[0,0], 2)
        self.assertEqual(conf_matrix.iloc[0,1], 1)
        self.assertEqual(conf_matrix.iloc[1,0], 0)
        self.assertEqual(conf_matrix.iloc[1,1], 3)
