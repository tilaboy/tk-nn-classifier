import spacy
import random
from pathlib import Path

from spacy.util import minibatch, compounding
from ..data_loader import get_spacy_data
from .. import LOGGER
from .utils import TrainHelper

class SpaceClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']

    def build_and_train(self):
        self.build_graph()
        train_data=get_spacy_data(
            self.config['datasets']['train'],
            shuffle=True,
            train_mode=True
        )
        eval_data=get_spacy_data(self.config['datasets']['eval'])
        self.train(train_data, eval_data)
        if 'test' in self.config['datasets']:
            self.evaluate_on_tests()

    def evaluate_on_tests(self):
        textcat = self.model.get_pipe("textcat")
        for test_set in  self.config['datasets']['test']:
            test_data = get_spacy_data(self.config['datasets']['test'][test_set])
            texts, cats = zip(*test_data)
            predicted_classes = list(self.predict_batch(texts))
            TrainHelper.eval_and_print(test_set_name, predicted_classes, lables)

    def build_graph(self):
        if self.config["spacy"]["model"] is not None:
            model = spacy.load(self.config["spacy"]["model"])
            LOGGER.info("Loaded model '%s'" % self.config["spacy"]["model"])
        else:
            model = spacy.blank(self.config["spacy"]["language"])  # create blank Language class
            LOGGER.info("Created blank '%s' model" % self.config["spacy"]["language"])

        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "textcat" not in model.pipe_names:
            textcat = model.create_pipe(
                "textcat",
                config={
                    "exclusive_classes": True,
                    "architecture": self.config["spacy"]["architecture"],
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
                scores = self.evaluate(eval_data)
                TrainHelper.print_progress(losses["textcat"], scores)
            self.confusion_matrix(eval_data)


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

    def load_saved_model(self, model_path==None):
        if model_path is None:
            model_path = self.config['model_path']
        self.model = spacy.load(model_path)

    def process_with_saved_model(self, input):
        result = self.model(input)
        doc = classifier(test_text)
        return [ doc.cats[label] for lable in uniq_lables ]

    def save(self, output_dir):
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            with self.model.use_params(self.optimizer.averages):
                self.model.to_disk(output_dir)
            print("Saved model to", output_dir)

    def predict_batch(self, texts):
        textcat = self.model.get_pipe("textcat")
        docs = (self.model.tokenizer(text) for text in texts)
        for doc in textcat.pipe(docs):
            yield doc.cats

    def split_train_test_data(self):
        """prepare data from our dataset."""
        train_data = list(get_spacy_data(self.config['train_data_path']))
        random.shuffle(train_data)
        texts, labels = zip(*train_data)
        cats = [{"yes": label == "yes", "no": label == "no"} for label in labels]
        split = int(len(train_data) * self.config['split_ratio'])

        return (
            list(zip(texts[:split], [{"cats": cats} for cats in cats[:split]])),
            list(zip(texts[split:], cats[split:]))
        )
