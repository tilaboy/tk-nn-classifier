"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
from tk_nn_classifier.classifiers.utils import TrainHelper, ConfusionMatrix

class ClassifierUtilTestCases(TestCase):
    """unit tests"""

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


        cm = ConfusionMatrix(eval, gold)
        self.assertEqual(cm.confusion_matrix[0][0], 2)
        self.assertEqual(cm.confusion_matrix[0][1], 1)
        self.assertEqual(cm.confusion_matrix[1][0], 0)
        self.assertEqual(cm.confusion_matrix[1][1], 3)

        cm_string = '========================================\n' + \
                    'gold\\eval  0          1         \n' + \
                    '0          2          1         \n' + \
                    '1          0          3         \n' + \
                    '========================================'

        self.assertEqual(str(cm), cm_string, 'comfusion matrix print format')
