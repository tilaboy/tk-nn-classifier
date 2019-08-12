import tensorflow as tf
from tensorflow import keras
import spacy

class Graph:
    def __init__(self, type):
        self.type = type

    def build_tf_graph(input_dimension=450, l_rate=0.02):
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

    def build_spacy_graph(model='en_core_web_sm'):
        if model is not None:
            nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")

        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "textcat" not in nlp.pipe_names:
            textcat = nlp.create_pipe(
                "textcat",
                config={
                    "exclusive_classes": True,
                    "architecture": "simple_cnn",
                }
            )
            nlp.add_pipe(textcat, last=True)
        # otherwise, get it, so we can add labels to it
        else:
            textcat = nlp.get_pipe("textcat")

        # add label to text classifier
        textcat.add_label("POSITIVE")
        textcat.add_label("NEGATIVE")

        return nlp
