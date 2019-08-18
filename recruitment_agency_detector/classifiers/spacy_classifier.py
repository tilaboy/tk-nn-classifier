import spacy
import random
from pathlib import Path

from spacy.util import minibatch, compounding
from ..utils.data_reader import get_data_from_trxml
from .. import LOGGER

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
            print("Created blank '{}' model".format(self.config["language"]))

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

        textcat.add_label("POSITIVE")
        textcat.add_label("NEGATIVE")


        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != "textcat"]

        with self.model.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.model.begin_training()
            if self.config.get('init_tok2vec', None) is not None:
                init_tok2vec = Path(self.config['init_tok2vec'])
                with init_tok2vec.open("rb") as file_:
                    textcat.model.tok2vec.from_bytes(file_.read())
            print("Training the model...")
            print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
            batch_sizes = compounding(4.0, 32.0, 1.001)
            for i in range(self.config['num_epochs']):
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
                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = self.evaluate_spacy(eval_data)
                print(
                    "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                        losses["textcat"],
                        scores["textcat_p"],
                        scores["textcat_r"],
                        scores["textcat_f"],
                    )
                )

    def prepare_train_test_data(self):
        """prepare data from our dataset."""
        train_data = get_data_from_trxml(self.config['train_data_path'])
        random.shuffle(train_data)
        texts, labels = zip(*train_data)
        cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]

        split = int(len(train_data) * self.config['split_ratio'])

        return (
            list(zip(texts[:split], [{"cats": cats} for cats in cats[:split]])),
            list(zip(texts[split:], cats[split:]))
        )


    def evaluate_spacy(self, eval_data):
        textcat = self.model.get_pipe("textcat")
        texts, cats = zip(*eval_data)
        docs = (self.model.tokenizer(text) for text in texts)
        tp = 0.0  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0  # True negatives
        for i, text in enumerate(texts):
            doc = self.model(text)
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

    def save(self, output_dir):
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            optimizer = self.model.begin_training()
            with self.model.use_params(optimizer.averages):
                self.model.to_disk(output_dir)
            print("Saved model to", output_dir)
