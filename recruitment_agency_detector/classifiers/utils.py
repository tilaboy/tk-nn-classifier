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
        model_path = min(model_names,
                         key=labmda x: int(os.stat(x).st_birthday)
                        )
        return model_path


class TrainHelper:
    def __init__(self):
        pass

    @staticmethod
    def print_progress_header():
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))

    @staticmethod
    def print_progress(loss, scores):
        print(
            "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
                loss,
                scores["precision"],
                scores["recall"],
                scores["f1"],
            )
        )

    @staticmethod
    def eval_and_print(test_set, predicted_classes, labels):
        scores = self._evaluate_score_on_class(predicted_classes, labels)
        self.print_test_score(test_set, scores)
        if type(predicted_classes[0]) == dict:
            cm = self._evaluate_confusion_matrix(predicted_classes, labels)
        else:
            cm = self._evaluate_confusion_matrix_binary_class(predicted_classes, lables)
        LOGGER.info("Confusion matrix:")
        print(cm)


    @staticmethod
    def _print_test_score(test_set_name, scores):
        print(
            "{0:^5}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
                test_set_name,
                scores["precision"],
                scores["recall"],
                scores["f1"],
            )
        )

    @staticmethod
    def _evaluate_score_on_class(eval, gold, target_cat=1):
        tp = 0.0  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0  # True negatives
        for i, cat in enumerate(eval):
            if cat == gold[i] == target_cat:
                tp += 1.0
            elif cat == target_cat and gold[i] != target_cat:
                fp += 1.0
            elif cat != target_cat and gold[i] != target_cat:
                tn += 1.0
            elif cat != target_cat and gold[i] == target_cat:
                fn += 1.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f_score}


    @staticmethod
    def _evaluate_score(eval, gold):
        tp = 0.0  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0  # True negatives
        for i, cats in enumerate(eval):
            for label, score in cats.items():
                if label not in gold[i]:
                    continue
                if label == "no":
                    continue
                if score >= 0.5 and gold[i][label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[i][label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[i][label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[i][label] >= 0.5:
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f_score}

    @staticmethod
    def _evaluate_confusion_matrix(eval, gold):
        categories = {label: i for i,label in enumerate(gold[0].keys())}
        gold_labels = _max_dict_value(gold)
        eval_labels = _max_dict_value(eval)

        cm = pd.crosstab(pd.Series(gold_labels, name='Actual'),
                         pd.Series(eval_labels, name='Predicted')
                         )
        return cm

    @staticmethod
    def _evaluate_confusion_matrix_binary_class(eval_labels, gold_labels):
        cm = pd.crosstab(pd.Series(gold_labels, name='Actual'),
                         pd.Series(eval_labels, name='Predicted')
                         )
        return cm

def _max_dict_value(cats_dicts):
    return [max(cats_dict, key=cats_dict.get) for cats_dict in cats_dicts]
