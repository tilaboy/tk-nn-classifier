import tensorflow as tf
from tensorflow import keras
import spacy
import random
from pathlib import Path

from spacy.util import minibatch, compounding

class Model:
    def __init__(self, type):
        self.type = type

    def build_graph(self, *args, **kwargs):
        if self.type == 'tf':
            self.model = self.build_tf_graph(*args, **kwargs)
        elif self.type == 'spacy':
            self.model = self.build_spacy_graph(*args, **kwargs)

    def train(self, *args, **kwargs):
        if self.type == 'tf':
            self.train_tf(*args, **kwargs)
        elif self.type == 'spacy':
            self.train_spacy(*args, **kwargs)

    def build_tf_graph(self, input_dimension=450, l_rate=0.02):
        """
        A hard coded training graph

        params:
            - input_dimension: the dimension of the input data
            - l_rate: the learning rate

        output: neural netword model
        """

        model = keras.Sequential()
        model.add(keras.layers.Dense(128, input_dim=input_dimension, activation=tf.nn.sigmoid))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(64, activation=tf.nn.sigmoid))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(l_rate),
                      metrics=['accuracy'])
        return model

    def train_tf(self, x_train, y_train, x_test, y_test):
        """Training process"""

        self.model.fit(
            x_train, y_train,
            epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(x_test, y_test)
        )


    def build_spacy_graph(self, pre_model=None):
        if pre_model is not None:
            model = spacy.load(pre_model)  # load existing spaCy model
            print("Loaded model '%s'" % pre_model)
        else:
            model = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")

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
        return model


    def train_spacy(self, train_texts, train_cats, eval_texts, eval_cats, n_iter=2, init_tok2vec=None):
        textcat = self.model.get_pipe("textcat")

        textcat.add_label("POSITIVE")
        textcat.add_label("NEGATIVE")

        train_data=list(zip(train_texts, train_cats))
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != "textcat"]
        with self.model.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.model.begin_training()
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
                    self.model.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = self.evaluate_spacy(eval_texts, eval_cats)
                print(
                    "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                        losses["textcat"],
                        scores["textcat_p"],
                        scores["textcat_r"],
                        scores["textcat_f"],
                    )
                )


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

    def save_model(self, output_dir):
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            optimizer = self.model.begin_training()
            with self.model.use_params(optimizer.averages):
                self.model.to_disk(output_dir)
            print("Saved model to", output_dir)
