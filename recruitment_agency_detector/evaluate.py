import tensorflow as tf
from tensorflow import keras
from xml_miner.miner import TRXMLMiner

from data_utils import DataReader
from recruitment_agency_detector import LOGGER

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


def compute_score(nlp, texts, labels):
    textcat = nlp.get_pipe("textcat")
    optimizer = nlp.begin_training()
    with textcat.model.use_params(optimizer.averages):
        scores = evaluate(nlp.tokenizer, textcat, texts, cats)
    print(
        "{0:.3f}\t{1:.3f}\t{2:.3f}".format(  # print a simple table
            scores["textcat_p"],
            scores["textcat_r"],
            scores["textcat_f"],
        )
    )


def evaluate_trxml_batch(model_path, data_path):
    nlp = spacy.load(model_path)
    train_data = get_data(data_path)
    train_texts, train_labels = zip(*train_data)
    print('compute scores on training set')
    compute_score(nlp, train_texts, train_labels)


def predict_trxml_batch(model_dir='first_model', output_file='result.txt'):
    print("Loading from", model_dir)
    nlp = spacy.load(model_dir)

    fh_output = open(output_file, 'w')
    fh_output.write('id\torg_name\tsite\tnew_predict\told_predict\turl\tscore\n')
    for test_text, category, id, orgname, site, url in get_data_with_details('data/random_trxmls'):
        test_text = prepare_input_text(text)
        doc = nlp(test_text)
        predict_cat = 1 if doc.cats['POSITIVE'] > doc.cats['NEGATIVE'] else 0
        fh_output.write(f"{id}\t{orgname}\t{site}\t{predict_cat}\t{category}\t{url}\t{doc.cats}\n")
    fh_output.close()
