from typing import List, Dict
import os
import platform
from tabulate import tabulate
from .. import LOGGER


def _file_creation_date(path_to_file: str) -> float:
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


def last_modified_folder(model_path):
    model_files = [
        os.path.join(model_path, name)
        for name in os.listdir(model_path)
    ]
    return max(model_files, key=lambda x: int(_file_creation_date(x)))


def eval_accuracy(predictions: List, gold_labels: List) -> float:
    '''
    func:
        - compute the accuracy

    params:
        - predictions
        - gold_labels

    output:
        - accuracy
    '''
    nr_right = nr_error = 0
    for pred, gold in zip(predictions, gold_labels):
        if pred == gold:
            nr_right += 1
        else:
            nr_error += 1
    return 0.0 if nr_right == 0 else (1.0 * nr_right) / (nr_right + nr_error)


def eval_precision_recall(predictions: List, gold_labels: List) -> Dict:
    '''
    input:
        - predictions
        - gold_labels

    func:
        - compute the precision and recall

    output:
        - precision
        - recall
        - f1
    '''

    uniq_labels = set(predictions + gold_labels)
    epsilon = 1e-8
    scores = {label: {'tp': 0, 'fp': epsilon, 'fn': epsilon}
              for label in uniq_labels}

    for pred, gold in zip(predictions, gold_labels):
        if pred == gold:
            scores[pred]['tp'] = scores[pred]['tp'] + 1.0
        else:
            scores[pred]['fp'] = scores[pred].get('fp', 1e-7) + 1.0
            scores[gold]['fn'] = scores[pred].get('fn', 1e-7) + 1.0

    for label in uniq_labels:
        scores[label]['precision'] = scores[label]['tp'] / (
            scores[label]['tp'] + scores[label]['fp'])
        scores[label]['recall'] = scores[label]['tp'] / (
            scores[label]['tp'] + scores[label]['fn'])

        if scores[label]['tp'] == 0:
            scores[label]['f1'] = 0.0
        else:
            prec_reca_sum = scores[label]['precision'] + scores[label]['recall']
            prec_reca_mul = scores[label]['precision'] * scores[label]['recall']
            scores[label]['f1'] = 2 * prec_reca_mul / prec_reca_sum
    return scores


def eval_confusion_matrix(predictions: List, gold_labels: List) -> List:
    """
    Generate a confusion matrix for multiple classification

    params:
        - eval: a list of integers or strings of predicted classes
        - gold: a list of integers or strings of known classes

    output:
        - confusion_matrix: 2-dimensional list of pairwise counts
    """

    cats = sorted(set(predictions + gold_labels))
    confusion_matrix = [['gold\\eval'] + cats]
    for cat in cats:
        confusion_matrix.append([cat] + [0] * len(cats))
    cat_id_map = {cat: id + 1 for id, cat in enumerate(cats)}
    for pred, gold in zip(predictions, gold_labels):
        row_id = cat_id_map[gold]
        col_id = cat_id_map[pred]
        confusion_matrix[row_id][col_id] += 1
    return confusion_matrix


def eval_predictions(predictions: List, gold_labels: List) -> None:
    # convert to str if the labels are not in the str format:
    predictions = [str(value) for value in predictions]
    gold_labels= [str(value) for value in gold_labels]

    accuracy = round(eval_accuracy(predictions, gold_labels), 3)
    print ('- overal accuracy: {:0.2f}'.format(accuracy))

    scores = eval_precision_recall(predictions, gold_labels)
    score_fields = ['precision', 'recall', 'f1']
    score_table = [
        [label] + [round(scores[label][field], 3) for field in score_fields]
        for label in sorted(scores.keys())
    ]
    print('- precision, recall and f1 scores:')
    print(tabulate(score_table, ['label'] + score_fields, missingval="-", tablefmt="github"))

    confusion_matrix = eval_confusion_matrix(predictions, gold_labels)
    print('- confusion matrix:')
    print(tabulate(confusion_matrix, headers="firstrow", missingval="-", tablefmt="github"))
