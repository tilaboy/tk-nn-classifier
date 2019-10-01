import os
import numpy
import pandas as pd
from .. import LOGGER

class FileHelper:
    def __init__(self):
        pass

    @staticmethod
    def last_modified_folder(model_path):
        model_names = [
                os.path.join(model_path, name)
                for name in os.listdir(model_path)
        ]
        model_path = max(model_names,
                         key=lambda x: int(os.stat(x).st_birthtime)
                        )
        return model_path


class TrainHelper:
    def __init__(self):
        pass

    @staticmethod
    def print_progress_header():
        print("{:^5}\t{:^5}".format("LOSS", "ACCU"))

    @staticmethod
    def print_progress(loss, accu):
        print("{0:.3f}\t{1:.3f}".format(loss,accu))


    @staticmethod
    def print_test_result(eval, gold):
        scores = TrainHelper._evaluate_f1_score(eval, gold)

        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("lable", "Prec", "Reca", "F1"))
        for label in scores:
            print(
                "{0:^5}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
                    label,
                    scores[label]["precision"],
                    scores[label]["recall"],
                    scores[label]["f1"]
                )
            )
        cm = TrainHelper._evaluate_confusion_matrix(eval, gold)
        LOGGER.info("Confusion matrix:")
        print(cm)

    @staticmethod
    def accuracy (eval, gold):
        correct = 0
        wrong = 0
        for i, cat in enumerate(eval):
            if cat == gold[i]:
                correct += 1
            else:
                wrong += 1
        return (1.0 * correct) / (correct + wrong)

    @staticmethod
    def max_dict_value(cats_dicts):
        return [
            max(cats_dict, key=cats_dict.get)
            for cats_dict in cats_dicts
        ]

    @staticmethod
    def _evaluate_f1_score(eval, gold):
        uniq_lables = set(eval + gold)
        scores = {}
        for label in uniq_lables:
            tp = 0.0  # True positives
            fp = 1e-8  # False positives
            fn = 1e-8  # False negatives
            tn = 0.0  # True negatives
            for i, cat in enumerate(eval):
                if label not in [cat, gold[i]]:
                    continue
                if cat == gold[i] == label:
                    tp += 1.0
                elif cat == label and gold[i] != label:
                    fp += 1.0
                elif cat != label and gold[i] == label:
                    fn += 1.0
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if (precision + recall) == 0:
                f_score = 0.0
            else:
                f_score = 2 * (precision * recall) / (precision + recall)
            scores[label] = {"precision": precision, "recall": recall, "f1": f_score}
        return scores



    @staticmethod
    def _evaluate_confusion_matrix(eval_labels, gold_labels):
        cm = pd.crosstab(pd.Series(gold_labels, name='Actual'),
                         pd.Series(eval_labels, name='Predicted')
                         )
        return cm
