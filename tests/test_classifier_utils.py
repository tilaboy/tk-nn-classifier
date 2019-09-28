"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
from recruitment_agency_detector.classifiers.utils import _max_dict_value
from recruitment_agency_detector.classifiers.utils import TrainHelper

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
        self.assertEqual(_max_dict_value(cats_dict), ['a', 'c', 'b'])


    def test_score(self):
        eval = [
            {'yes': 0.7,'no': 0.3},
            {'yes': 0.9,'no': 0.1},
            {'yes': 0.5,'no': 0.5},
            {'yes': 0.3,'no': 0.7},
            {'yes': 0.8,'no': 0.2},
            {'yes': 0.1,'no': 0.9}
        ]

        gold = [
            {'yes': 1.0,'no': 0.0},
            {'yes': 0.0,'no': 1.0},
            {'yes': 1.0,'no': 0.0},
            {'yes': 0.0,'no': 1.0},
            {'yes': 1.0,'no': 0.0},
            {'yes': 0.0,'no': 1.0}

        ]

        score = self.train_helper._evaluate_score(eval, gold)
        self.assertAlmostEqual(score["precision"], 0.75, places=2)
        self.assertAlmostEqual(score["recall"], 1.00, places=2)
        self.assertAlmostEqual(score["f1"], 0.857, places=2)


    def test_confusion(self):
        eval = [
            {'yes': 0.7,'no': 0.3},
            {'yes': 0.9,'no': 0.1},
            {'yes': 0.5,'no': 0.5},
            {'yes': 0.3,'no': 0.7},
            {'yes': 0.8,'no': 0.2},
            {'yes': 0.1,'no': 0.9}
        ]

        gold = [
            {'yes': 1.0,'no': 0.0},
            {'yes': 0.0,'no': 1.0},
            {'yes': 1.0,'no': 0.0},
            {'yes': 0.0,'no': 1.0},
            {'yes': 1.0,'no': 0.0},
            {'yes': 0.0,'no': 1.0}

        ]

        conf_matrix = self.train_helper._evaluate_confusion_matrix(eval, gold)
        self.assertEqual(conf_matrix.iloc[0,0], 2)
        self.assertEqual(conf_matrix.iloc[0,1], 1)
        self.assertEqual(conf_matrix.iloc[1,0], 0)
        self.assertEqual(conf_matrix.iloc[1,1], 3)

    def test_confusion_binary(self):
        eval = [1,1,1,0,1,0]
        gold = [1,0,1,0,1,0]

        conf_matrix = \
                self.train_helper._evaluate_confusion_matrix_binary_class(
                        eval, gold)
        self.assertEqual(conf_matrix.iloc[0,0], 2)
        self.assertEqual(conf_matrix.iloc[0,1], 1)
        self.assertEqual(conf_matrix.iloc[1,0], 0)
        self.assertEqual(conf_matrix.iloc[1,1], 3)
