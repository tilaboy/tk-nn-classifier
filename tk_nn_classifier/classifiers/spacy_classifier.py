import spacy
from pathlib import Path
import random
from spacy.util import minibatch, compounding

from ..model_input import SpacyDataReader
from .. import LOGGER
from ..exceptions import ConfigError
from .base_classifier import BaseClassifier
from .utils import eval_predictions, eval_accuracy


class SpacyClassifier(BaseClassifier):
    def __init__(self, config):
        super().__init__(config)
        self.data_reader = SpacyDataReader(self.config)

    def build_and_train(self, train_data, eval_data):
        self.build_graph()
        self.train(train_data, eval_data)
        self.save(self.config['model_path'])
        if 'test' in self.config['datasets']:
            self.evaluate_on_sets()

    def prepare_input(self, data_set, train_mode):
        return self.data_reader.model_input(data_set, train_mode)

    def build_graph(self):
        if self.config["spacy"]["model"] is not None:
            # load pretrained spaCy model
            model = spacy.load(self.config["spacy"]["model"])
            LOGGER.info("Loaded model '%s'" % self.config["spacy"]["model"])
        else:
            # create blank Language class
            model = spacy.blank(self.config["spacy"]["language"])
            LOGGER.info("Created blank '%s' model" %
                        self.config["spacy"]["language"])

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
        self.model = model

    def train(self, train_data, eval_data):
        textcat = self.model.get_pipe("textcat")
        for label in self.data_reader.label_mapper.label_to_classid:
            textcat.add_label(label)

        # get names of other pipes to disable them during training
        other_pipes = [
            pipe for pipe in self.model.pipe_names
            if pipe != "textcat"
        ]

        with self.model.disable_pipes(*other_pipes):  # only train textcat
            self.optimizer = self.model.begin_training()
            if self.config.get('init_tok2vec', None) is not None:
                init_tok2vec = Path(self.config['init_tok2vec'])
                with init_tok2vec.open("rb") as file_:
                    textcat.model.tok2vec.from_bytes(file_.read())
            LOGGER.info("Training the model...")
            batch_sizes = compounding(4.0, 32.0, 1.001)

            LOGGER.info("{:^5}\t{:^5}".format("LOSS", "ACCU"))
            for i in range(self.config['num_epochs']):
                losses = self._update_one_epoch(train_data, batch_sizes)

                # eval set accu
                texts, cats = zip(*eval_data)
                _accuracy = eval_accuracy(
                    self.classify_batch(texts),
                    (max(cat, key=cat.get) for cat in cats)
                )

                # print progress
                LOGGER.info("{0:.3f}\t{1:.3f}".format(losses["textcat"],
                                                      _accuracy))

    def _update_one_epoch(self, train_data, batch_sizes):
        losses = {}
        # batch up the examples using spaCy's minibatch
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, cats = zip(*batch)
            self.model.update(
                              texts,
                              cats,
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
        LOGGER.info("Saved model to %s" % output_dir)
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            self.model.to_disk(output_dir)
        else:
            raise ValueError('output_dir is not set')


    def classify_batch(self, texts):
        for cat in self.predict_likelihoods(texts):
            yield max(cat, key=cat.get)

    def predict_likelihoods(self, texts):
        textcat = self.model.get_pipe("textcat")
        docs = (self.model.tokenizer(text) for text in texts)
        for doc in textcat.pipe(docs):
            yield doc.cats
