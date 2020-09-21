import os
import platform
import pandas as pd
from .. import LOGGER


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


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
                         key=lambda x: int(creation_date(x)))
        return model_path


class TrainHelper:
    def __init__(self):
        pass

    @staticmethod
    def print_progress_header():
        print("{:^5}\t{:^5}".format("LOSS", "ACCU"))

    @staticmethod
    def print_progress(loss, accu):
        print("{0:.3f}\t{1:.3f}".format(loss, accu))

    @staticmethod
    def print_test_result(eval, gold):
        scores = TrainHelper._evaluate_f1_score(eval, gold)

        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("label",
                                                  "Prec",
                                                  "Reca",
                                                  "F1"))
        for label in scores:
            print("{0:^5}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
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
    def accuracy(eval, gold):
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
        uniq_labels = set(eval)
        uniq_labels.union(set(gold))
        scores = {}
        for label in uniq_labels:
            tp = 0.0  # True positives
            fp = 1e-8  # False positives
            fn = 1e-8  # False negatives
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
            scores[label] = {"precision": precision,
                             "recall": recall,
                             "f1": f_score}
        return scores

    @staticmethod
    def _evaluate_confusion_matrix(eval, gold):
        """
        Generate a confusion matrix for multiple classification

        params:
            - eval: a list of integers or strings of predicted classes
            - gold: a list of integers or strings of known classes
        output:
            - confu_matrix: 2-dimensional list of pairwise counts
        """
        def _to_str(arr):
            return [str(value) for value in arr]

        gold = _to_str(gold)
        eval = _to_str(eval)
        cats = sorted(set(gold + eval))
        min_len = max([len(cat) for cat in cats]) + 2
        cellwidth = max([min_len, 10])

        confu_matrix = [[0 for _ in cats] for _ in cats]
        cat_id_map = {cat:id for id, cat in enumerate(cats)}
        for predicted, real in zip(eval, gold):
            confu_matrix[cat_id_map[real]][cat_id_map[predicted]] += 1

        sep_line = "=" * (cellwidth * (len(cats) + 2))
        cm_table = sep_line + "\n"

        table_header = " ".join([("{}".format(cat)).ljust(cellwidth) for cat in cats])
        cm_table += "gold\\eval".ljust(cellwidth + 1) + table_header + "\n"
        for cat, row in zip(cats, confu_matrix):
            row_start = ("{}".format(cat)).ljust(cellwidth)
            row_string = " ".join([str(val).ljust(cellwidth) for val in row])
            cm_table += "{} {}\n".format(row_start, row_string)
        cm_table += sep_line
        return cm_table
