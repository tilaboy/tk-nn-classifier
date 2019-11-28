import os
import spacy
import json
from pathlib import Path
import random
from spacy.util import minibatch, compounding

from ..data_loader import SpacyDataReader
from .. import LOGGER
from .utils import TrainHelper

class SpaceClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']
        self.data_reader = SpacyDataReader(self.config)

    def build_and_train(self):
        self.build_graph()

        if 'all_data' in self.config['datasets']:
            if 'train' in self.config['datasets'] or \
            'eval' in self.config['datasets']:
                raise ValueError("config conflict: all_data <=> train/eval")
            else:
                # split the data
                LOGGER.info('split all_data into train and test')
                train_source, eval_source = self.data_reader.get_split_data()
                self.config['datasets']['train'] = train_source
                self.config['datasets']['eval'] = eval_source

        train_data=self.data_reader.get_data(
            self.config['datasets']['train'],
            shuffle=True,
            train_mode=True
        )
        eval_data=self.data_reader.get_data(self.config['datasets']['eval'])

        self.train(train_data, eval_data)

        self.save(self.config['model_path'])

        if 'test' in self.config['datasets']:
            self.evaluate_on_tests()

    def build_graph(self):
        if self.config["spacy"]["model"] is not None:
            # load pretrained spaCy model
            model = spacy.load(self.config["spacy"]["model"])
            LOGGER.info("Loaded model '%s'" % self.config["spacy"]["model"])
        else:
            # create blank Language class
            model = spacy.blank(self.config["spacy"]["language"])
            LOGGER.info("Created blank '%s' model" % self.config["spacy"]["language"])

        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "textcat" not in model.pipe_names:
            textcat = model.create_pipe(
                "textcat",
                config={
                    "exclusive_classes": True,
                    "architecture": self.config["spacy"]["arch"],
                }
            )
            model.add_pipe(textcat, last=True)
        self.model =  model

    def train(self, train_data, eval_data):
        textcat = self.model.get_pipe("textcat")
        for label in self.data_reader.label_mapper.label_to_classid:
            textcat.add_label(label)

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != "textcat"]

        with self.model.disable_pipes(*other_pipes):  # only train textcat
            self.optimizer = self.model.begin_training()
            if self.config.get('init_tok2vec', None) is not None:
                init_tok2vec = Path(self.config['init_tok2vec'])
                with init_tok2vec.open("rb") as file_:
                    textcat.model.tok2vec.from_bytes(file_.read())
            LOGGER.info("Training the model...")
            TrainHelper.print_progress_header()
            batch_sizes = compounding(4.0, 32.0, 1.001)

            for i in range(self.config['num_epochs']):
                losses = self._update_one_epoch(train_data, batch_sizes)
                pred, gold = self.evaluate(eval_data, 'train', losses["textcat"])


    def _update_one_epoch(self, train_data, batch_sizes):
        losses = {}
        # batch up the examples using spaCy's minibatch
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, annotations = zip(*batch)
            self.model.update(
                              texts,
                              annotations,
                              sgd=self.optimizer,
                              drop=self.config["dropout_rate"],
                              losses=losses
                              )
        return losses

    def load_saved_model(self, model_path=None):
        if model_path is None:
            model_path = self.config['model_path']
        self.model = spacy.load(model_path)

    def save(self, output_dir):
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            with self.model.use_params(self.optimizer.averages):
                self.model.to_disk(output_dir)
            LOGGER.info("Saved model to %s" % output_dir)

    def process_with_saved_model(self, input):
        result = self.model(input)
        return result.cats

    def evaluate_on_tests(self):
        train_helper = TrainHelper()
        for test_set in self.config['datasets']['test']:
            LOGGER.info('test_set: %s' % test_set)
            test_data = self.data_reader.get_data(
                    self.config['datasets']['test'][test_set])
            eval, gold = self.evaluate(test_data, 'test')

    def evaluate(self, dataset, mode='train', losses=0):
        eval, gold = self.prediction_on_set(dataset)
        if mode == 'test':
            TrainHelper.print_test_result(eval, gold)
        elif mode == 'train':
            accu = TrainHelper.accuracy(eval, gold)
            TrainHelper.print_progress(losses, accu)
        return eval, gold

    def prediction_on_set(self, dataset):
        texts, cats = zip(*dataset)
        predicted_prob = list(self.predict_batch(texts))
        gold_classes = TrainHelper.max_dict_value(cats)
        predicted_classes = TrainHelper.max_dict_value(predicted_prob)
        return predicted_classes, gold_classes

    def predict_batch(self, texts):
        textcat = self.model.get_pipe("textcat")
        docs = (self.model.tokenizer(text) for text in texts)
        for doc in textcat.pipe(docs):
            yield doc.cats
