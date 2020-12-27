"""unit tests for classifier utils functions"""
import os
from unittest import TestCase
from tk_nn_classifier.classifiers.utils import eval_accuracy
from tk_nn_classifier.classifiers.utils import eval_precision_recall
from tk_nn_classifier.classifiers.utils import eval_confusion_matrix


class ClassifierUtilBinaryTestCases(TestCase):
    """unit tests"""
    def setUp(self):
        self.eval = ['yes','yes','yes','no','yes','no']
        self.gold = ['yes','no','yes','no','yes','no']

    def test_accuracy_binary(self):
        self.assertAlmostEqual(eval_accuracy(self.eval, self.gold),
                               0.83,
                               places=2,
                               msg='basic binary accuracy')


        self.assertAlmostEqual(eval_accuracy(self.eval, ['yes'] * 6),
                               0.67,
                               places=2,
                               msg='gold all yes binary accuracy')

        self.assertAlmostEqual(eval_accuracy(['yes'] * 6, self.gold),
                               0.5,
                               places=2,
                               msg='eval all yes binary accuracy')

        self.assertAlmostEqual(eval_accuracy(['yes'] * 6, ['no'] * 6),
                               0.0,
                               places=2,
                               msg='all wrong binary accuracy')

    def test_precision_recall_binary(self):
        expt_scores = {
            'yes': {'precision': 0.75, 'recall': 1.0, 'f1': 0.86},
            'no': {'precision': 1.0, 'recall': 0.67, 'f1': 0.80},
        }
        pred_scores = eval_precision_recall(self.eval, self.gold)

        for label in expt_scores:
            for field in ['precision', 'recall', 'f1']:
                self.assertAlmostEqual(
                    pred_scores[label][field],
                    expt_scores[label][field],
                    places=2,
                    msg='label {} field {}'.format(label, field))

    def test_precision_recall_all_yes_binary(self):
        expt_scores = {
            'yes': {'precision': 0.5, 'recall': 1.0, 'f1': 0.67},
            'no': {'precision': 0, 'recall': 0, 'f1': 0},
        }
        pred_scores = eval_precision_recall(['yes'] * 6, self.gold)

        for label in expt_scores:
            for field in ['precision', 'recall', 'f1']:
                self.assertAlmostEqual(
                    pred_scores[label][field],
                    expt_scores[label][field],
                    places=2,
                    msg='label {} field {}'.format(label, field))


    def test_confusion_binary(self):
        eval = ['yes','yes','yes','no','yes','no']
        gold = ['yes','no','yes','no','yes','no']
        expt_cm = [['gold\\eval', 'no', 'yes'], ['no', 2, 1], ['yes', 0, 3]]
        self.assertEqual(eval_confusion_matrix(eval, gold), expt_cm)
