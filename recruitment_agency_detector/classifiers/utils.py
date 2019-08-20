import numpy
from .. import LOGGER

class Evaluater:
    def __init__(self, type):
        self.type = type

    def tf_evaluation(model, test_file, data_reader):
        """Evaluate on the data set"""

        text_lines, x_eval, y_eval = data_reader.read_file(test_file)
        results = model.predict(x_eval)
        for text, i in enumerate(text_lines):
            LOGGER.info("predicted={:0.2f}\tclass={}\ttext={}".format(
                results[i][0], y_eval[i], text))


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
    def evaluate_score(eval, gold):
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
    def confusion_matrix(eval, gold):
        categories = {lable: i for i,label in enumerate(gold[0].keys())}
        gold_labels = max_dict_value(gold)
        eval_labels = max_dict_value(eval)


        cm = numpy.zeros((length(categories), length(categories)))
        for eval_label, gold_label in zip(eval_labels, gold_labels):
            cm[categories[eval_label]][categories[gold_label]] += 1

        print(cm)

def max_dict_value(cats_dicts):
    return [max(item, key=item.get) for cats_ict in cats_dicts]


def predict_trxml_batch(model_dir='first_model', output_file='result.txt'):
    print("Loading from", model_dir)
    nlp = spacy.load(model_dir)

    fh_output = open(output_file, 'w')
    fh_output.write('id\torg_name\tsite\tnew_predict\told_predict\turl\tscore\n')
    for test_text, category, id, orgname, site, url in get_data_with_details('data/random_trxmls'):
        test_text = prepare_input_text(text)
        doc = nlp(test_text)
        predict_cat = 1 if doc.cats['yes'] > doc.cats['no'] else 0
        fh_output.write(f"{id}\t{orgname}\t{site}\t{predict_cat}\t{category}\t{url}\t{doc.cats}\n")
    fh_output.close()
