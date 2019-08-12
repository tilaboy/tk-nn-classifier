'''Defines the Neural Network model used for skill validation'''
import random

import tensorflow as tf
from tensorflow import keras
from spacy.util import minibatch, compounding

from data_utils import DataReader
from recruitment_agency_detector import LOGGER


class Trainer:
    def __init__(type):
        self.type = type

    def train_tf(self, x_train, y_train, x_test, y_test, model, config):
        """Training process"""

        model.fit(
            x_train, y_train,
            epochs=config['num_epochs'], batch_size=config['batch_size'],
            validation_data=(x_test, y_test)
        )
        return model

    def train_spacy(self, nlp, train_data, eval_texts, eval_cats, n_iter=20, init_tok2vec=None):
        textcat = nlp.get_pipe("textcat")

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
        with nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = nlp.begin_training()
            if init_tok2vec is not None:
                with init_tok2vec.open("rb") as file_:
                    textcat.model.tok2vec.from_bytes(file_.read())
            print("Training the model...")
            print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
            batch_sizes = compounding(4.0, 32.0, 1.001)
            for i in range(n_iter):
                losses = {}
                # batch up the examples using spaCy's minibatch
                random.shuffle(train_data)
                batches = minibatch(train_data, size=batch_sizes)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = self.evaluate_spacy(nlp.tokenizer, textcat, eval_texts, eval_cats)
                print(
                    "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                        losses["textcat"],
                        scores["textcat_p"],
                        scores["textcat_r"],
                        scores["textcat_f"],
                    )
                )

        return nlp

    def evaluate_spacy(self, tokenizer, textcat, texts, cats):
        docs = (tokenizer(text) for text in texts)
        tp = 0.0  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0  # True negatives
        for i, doc in enumerate(textcat.pipe(docs)):
            gold = cats[i]
            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if label == "NEGATIVE":
                    continue
                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


def build_model(model_cfg):
    """The train, save and evaluation if needed"""

    LOGGER.info('Loading: %s', model_cfg['embedding_file'])
    data_reader = DataReader(model_cfg['embedding_file'])

    LOGGER.info('Reading: %s', model_cfg['train_data'])
    _, x_train, y_train = data_reader.read_file(model_cfg['train_data'])

    LOGGER.info('Reading: %s', model_cfg['test_data'])
    _, x_test, y_test = data_reader.read_file(model_cfg['test_data'])

    input_dimension = x_train.shape[1]
    LOGGER.info('Input dimension: %i', input_dimension)

    model = build_tf_graph(input_dimension)

    LOGGER.info("Start training")
    model = train_tf(x_train, y_train, x_test, y_test, model, model_cfg)
    model.save(model_cfg['model_file'])

    test_loss, test_acc = model.evaluate(x_test, y_test)
    LOGGER.info("Test: loss {}\tacc {}".format(test_loss, test_acc))

    if 'devel_data' in model_cfg:
        _, x_devel, y_devel = data_reader.read_file(model_cfg['devel_data'])
        dev_loss, dev_acc = model.evaluate(x_devel, y_devel)
        LOGGER.info("Devel: loss {}\tacc {}".format(dev_loss, dev_acc))
