'''Defines the Neural Network model used for skill validation'''
import random

import tensorflow as tf
from tensorflow import keras
from spacy.util import minibatch, compounding

from recruitment_agency_detector import LOGGER


class Trainer:
    def __init__(config, model):
        self.config = config
        self.model = model

    def train_tf(self, x_train, y_train, x_test, y_test):
        """Training process"""

        self.model.fit(
            x_train, y_train,
            epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(x_test, y_test)
        )
        return self.model

    def train_spacy(self, train_data, eval_texts, eval_cats, n_iter=20, init_tok2vec=None):
        textcat = self.model.get_pipe("textcat")

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != "textcat"]
        with self.model.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.model.begin_training()
            if init_tok2vec is not None:
                with init_tok2vec.open("rb") as file_:
                    textcat.nlp.tok2vec.from_bytes(file_.read())
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
                    self.model.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
                with textcat.nlp.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = evaluate_spacy(, textcat, eval_texts, eval_cats)
                print(
                    "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                        losses["textcat"],
                        scores["textcat_p"],
                        scores["textcat_r"],
                        scores["textcat_f"],
                    )
                )

        return model

    def evaluate_spacy(self, texts, cats):
        textcat = self.model.get_pipe("textcat")
        docs = (self.model.tokenizer(text) for text in texts)
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
