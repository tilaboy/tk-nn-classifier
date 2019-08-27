import os
import tensorflow as tf
from tensorflow import keras
from tensorboard import summary as summary_lib
import numpy as np
import random
from pathlib import Path
from tk_preprocessing.common_processor import char_normalization
from tensorflow.python.keras.preprocessing import sequence
from ..data_loader import WordVector
from ..data_loader.trxml_reader import get_tf_data, tokenize

class TFClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']
        self.max_sequence_length = config['max_sequence_length']

    def build_and_train(self):
        self.load_embedding()
        self.build_graph()
        self.train()

    def load_embedding(self):
        self.embedding = WordVector(self.config['embedding_file'])

    def load_data_set(self, data_path):
        data_set = get_tf_data(data_path)
        texts, labels = zip(*data_set)

        data_ids = [ [
                self.embedding.get_index(token)
                for token in tokenize(text)]
                for text in texts ]
        data_length = np.array([
            min(len(data_id), self.max_sequence_length)
            for data_id in data_ids
        ])

        data = sequence.pad_sequences(data_ids,
                                 maxlen=self.max_sequence_length,
                                 truncating='post',
                                 padding='post',
                                 value=WordVector.PAD_ID)
        return (data, np.array(labels), data_length)



    @staticmethod
    def parser(x, length, y):
        features = {"x": x, "len": length}
        return features, y

    def train_input_fn(self):
        (data, labels, data_length) = self.load_data_set(self.config['datasets']['train'])

        dataset = tf.data.Dataset.from_tensor_slices((data, data_length, labels))
        dataset = dataset.shuffle(buffer_size=len(data))
        dataset = dataset.batch(self.config['batch_size'])
        dataset = dataset.map(self.parser)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def eval_input_fn(self):
        (data, labels, data_length) = self.load_data_set(self.config['datasets']['eval'])

        dataset = tf.data.Dataset.from_tensor_slices((data, data_length, labels))
        dataset = dataset.batch(self.config['batch_size'])
        dataset = dataset.map(self.parser)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


    def build_graph(self):
        def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
            assert dtype is tf.float32
            return self.embedding.vectors

        #params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}
        params = {'embedding_initializer': my_initializer}
        self.classifier = tf.estimator.Estimator(
                model_fn=self.cnn_model_fn,
                model_dir=os.path.join(self.config['model_path'], 'cnn'),
                params=params
        )



    def cnn_model_fn(self, features, labels, mode, params):
        """
        cnn model

        params:
            - feature: the vector index of the input token sequence
            - labels: the training labels
            - mode: session model, train, eval, or predict
            - params: other parameters, e.g. the initializer of Word Embedding

        output: neural netword model
        """

        input_layer = tf.contrib.layers.embed_sequence(
            features['x'],
            self.embedding.vocab_size,
            self.embedding.vector_size,
            initializer=params['embedding_initializer'],
            trainable=False
        )

        training = mode == tf.estimator.ModeKeys.TRAIN
        dropout_emb = tf.layers.dropout(inputs=input_layer,
                                        rate=self.config['dropout_rate'],
                                        training=training)

        conv = tf.layers.conv1d(
            inputs=dropout_emb,
            filters=32,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        # Global Max Pooling
        pool = tf.reduce_max(input_tensor=conv, axis=1)

        hidden = tf.layers.dense(inputs=pool, units=250, activation=tf.nn.relu)

        dropout_hidden = tf.layers.dropout(inputs=hidden,
                                           rate=0.2,
                                           training=training)

        logits = tf.layers.dense(inputs=dropout_hidden, units=1)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # This will be None when predicting
        if labels is not None:
            labels = tf.reshape(labels, [-1, 1])

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(labels, predictions['classes'])

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {"accuracy": accuracy}
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            tf.summary.scalar('my_accuracy', accuracy[1])

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    def train(self):
        """Training process"""
        # Save a reference to the classifier to run predictions later
        self.classifier.train(input_fn=self.train_input_fn, steps=self.config['num_epochs'])

        (x_test, y_test, y_length) = self.load_data_set(self.config['datasets']['eval'])

        eval_results = self.classifier.evaluate(input_fn=self.eval_input_fn)
        #predictions = np.array([p['classes'][0] for p in self.classifier.predict(input_fn=self.eval_input_fn)])
