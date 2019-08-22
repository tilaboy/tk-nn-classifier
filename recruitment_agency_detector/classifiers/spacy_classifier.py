import spacy
import random
from pathlib import Path

from spacy.util import minibatch, compounding
from ..data_loader import get_train_data
from .. import LOGGER
from .utils import TrainHelper

class SpaceClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']

    def build_and_train(self):
        self.build_graph()
        train_data, test_data = self.prepare_train_test_data()
        self.train(train_data, test_data)

    def build_graph(self):
        if self.config["spacy_model"] is not None:
            model = spacy.load(self.config["spacy_model"])  # load existing spaCy model
            LOGGER.info("Loaded model '%s'" % self.config["spacy_model"])
        else:
            model = spacy.blank(self.config["language"])  # create blank Language class
            LOGGER.info("Created blank '%s' model" % self.config["language"])

        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "textcat" not in model.pipe_names:
            textcat = model.create_pipe(
                "textcat",
                config={
                    "exclusive_classes": True,
                    "architecture": "simple_cnn",
                }
            )
            model.add_pipe(textcat, last=True)
        self.model =  model

    def train(self, train_data, eval_data):
        textcat = self.model.get_pipe("textcat")

        textcat.add_label("yes")
        textcat.add_label("no")

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != "textcat"]

        with self.model.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.model.begin_training()
            if self.config.get('init_tok2vec', None) is not None:
                init_tok2vec = Path(self.config['init_tok2vec'])
                with init_tok2vec.open("rb") as file_:
                    textcat.model.tok2vec.from_bytes(file_.read())
            LOGGER.info("Training the model...")
            TrainHelper.print_progress_header()
            batch_sizes = compounding(4.0, 32.0, 1.001)

            for i in range(self.config['num_epochs']):
                losses = self._update_one_epoch(train_data, batch_sizes, optimizer)
                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = self.evaluate(eval_data)

                TrainHelper.print_progress(losses["textcat"], scores)
            self.confusion_matrix(eval_data)

    def _update_one_epoch(self, train_data, batch_sizes, optimizer):
        losses = {}
        # batch up the examples using spaCy's minibatch
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, annotations = zip(*batch)
            self.model.update(
                              texts,
                              annotations,
                              sgd=optimizer,
                              drop=self.config["dropout_rate"],
                              losses=losses
                              )
        return losses

    def prepare_train_test_data(self):
        """prepare data from our dataset."""
        train_data = list(get_train_data(self.config['train_data_path']))
        random.shuffle(train_data)
        texts, labels = zip(*train_data)
        cats = [{"yes": label == "yes", "no": label == "no"} for label in labels]
        split = int(len(train_data) * self.config['split_ratio'])

        return (
            list(zip(texts[:split], [{"cats": cats} for cats in cats[:split]])),
            list(zip(texts[split:], cats[split:]))
        )

    def predict_batch(self, texts):
        textcat = self.model.get_pipe("textcat")
        docs = (self.model.tokenizer(text) for text in texts)
        for doc in textcat.pipe(docs):
            yield doc.cats


    def confusion_matrix(self, eval_data):
        texts, cats = zip(*eval_data)
        return TrainHelper.evaluate_confusion_matrix(self.predict_batch(texts), cats)


    def evaluate(self, eval_data):
        texts, cats = zip(*eval_data)
        return TrainHelper.evaluate_score(self.predict_batch(texts), cats)



    def save(self, output_dir):
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            optimizer = self.model.begin_training()
            with self.model.use_params(optimizer.averages):
                self.model.to_disk(output_dir)
            print("Saved model to", output_dir)
