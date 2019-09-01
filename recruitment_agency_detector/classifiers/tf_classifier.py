import os
import tensorflow as tf
from tensorflow import keras
from tensorboard import summary as summary_lib
import numpy as np
import random
from pathlib import Path
import functools
from tk_preprocessing.common_processor import char_normalization
from tensorflow.python.keras.preprocessing import sequence
from ..data_loader import WordVector
from ..data_loader.trxml_reader import get_tf_data, tokenize
from .. import LOGGER


class TFClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']
        self.max_sequence_length = config['max_sequence_length']
        self.data_sets = {}

    def build_and_train(self):
        self.load_embedding()
        self.build_graph()
        self.train()

    def load_embedding(self):
        self.embedding = WordVector(self.config['embedding_file'])

    def load_data_set(self, data_path):
        if data_path not in self.data_sets:
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
            self.data_sets[data_path] = (data, np.array(labels), data_length)
        return self.data_sets[data_path]


    @staticmethod
    def _data_parser(x, length, y):
        features = {"x": x, "len": length}
        return features, y

    def input_fn(self, data_path, shuffle_and_repeat=False ):
        LOGGER.info("load data from %s", data_path)
        (data, labels, data_length) = self.load_data_set(data_path)

        dataset = tf.data.Dataset.from_tensor_slices((data, data_length, labels))
        if shuffle_and_repeat:
            dataset = dataset.shuffle(buffer_size=len(data))
            dataset = dataset.repeat(self.config['num_epochs'])

        dataset = dataset.batch(self.config['batch_size'])
        dataset = dataset.map(self._data_parser)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


    def build_graph(self):
        def embedding_initializer(shape=None, dtype=tf.float32, partition_info=None):
            assert dtype is tf.float32
            return self.embedding.vectors

        #params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}
        params = {'embedding_initializer': embedding_initializer}

        self.model_dir = os.path.join(self.config['model_path'], 'cnn')

        run_config = tf.estimator.RunConfig(save_checkpoints_steps=100,
                                            save_summary_steps=100,
                                            model_dir=self.model_dir,
                                            keep_checkpoint_max=3)

        self.classifier = tf.estimator.Estimator(
                model_fn=self.cnn_model_fn,
                #config=run_config,
                model_dir=self.model_dir,
                params=params
        )

    def cnn_model_fn(self, features, labels, mode, params):
        training = mode == tf.estimator.ModeKeys.TRAIN

        input_layer = tf.contrib.layers.embed_sequence(
            features['x'],
            self.embedding.vocab_size,
            self.embedding.vector_size,
            initializer=params['embedding_initializer'],
            trainable=False
        )

        dropout_emb = tf.layers.dropout(inputs=input_layer,
                                        rate=self.config['dropout_rate'],
                                        training=training)

        # (batch, 512, 150) -> (batch, 256, 32)
        conv = tf.layers.conv1d(
            inputs=dropout_emb,
            filters=32,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        pool = tf.layers.max_pooling1d(inputs=conv, pool_size=2, strides=2, padding='same')
        flat = tf.contrib.layers.flatten(pool)
        dropout_flat = tf.layers.dropout(inputs=flat,
                                         rate=self.config['dropout_rate'],
                                         training=training)
        logits = tf.layers.dense(inputs=dropout_flat, units=2)

        #pool = tf.reduce_max(input_tensor=conv, axis=1)
        #hidden = tf.layers.dense(inputs=pool, units=250, activation=tf.nn.relu)
        #dropout_hidden = tf.layers.dropout(inputs=hidden,
        #                                   rate=0.2,
        #                                   training=training)
        #logits = tf.layers.dense(inputs=dropout_hidden, units=2)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:

            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predictions)}

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )

        # This will be None when predicting
        if labels is not None:
            onehot_labels = tf.one_hot(labels, 2, 1, 0)
            #labels = tf.reshape(labels, [-1, 1])

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        metric_ops = {
            "accuracy": tf.metrics.accuracy(labels, predictions['classes']),
            "auc": tf.metrics.auc(labels, predictions['classes'])
        }

        for metric_type, metric_op in metric_ops.items():
            # v[1] is the update op of the metrics object
            tf.summary.scalar(metric_type, metric_op[1])

        if mode == tf.estimator.ModeKeys.EVAL:

            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=metric_ops)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:

            optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))


    def multi_layer_cnn_model_fn(self, features, labels, mode, params):
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

        # (batch, 512, 150) -> (batch, 256, 32)
        conv_1 = tf.layers.conv1d(
            inputs=dropout_emb,
            filters=32,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        pool_1 = tf.layers.max_pooling1d(inputs=conv_1, pool_size=2, strides=2, padding='same')

        # (batch, 256, 32) -> (batch, 128, 32)
        conv_2 = tf.layers.conv1d(
            inputs=pool_1,
            filters=32,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        pool_2 = tf.layers.max_pooling1d(inputs=conv_2, pool_size=2, strides=2, padding='same')

        # (batch, 128, 32) -> (batch, 64, 32)
        conv_3 = tf.layers.conv1d(
            inputs=pool_2,
            filters=32,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        pool_3 = tf.layers.max_pooling1d(inputs=conv_3, pool_size=2, strides=2, padding='same')

        # fully connection layer
        #flat = tf.reshape(pool_3, (-1, 64*32))
        flat = tf.contrib.layers.flatten(pool_3)
        dropout_flat = tf.layers.dropout(inputs=flat,
                                         rate=self.config['dropout_rate'],
                                         training=training)

        #pool = tf.reduce_max(input_tensor=conv_3, axis=1)
        #hidden = tf.layers.dense(inputs=dropout_flat, units=128, activation=tf.nn.relu)
        #dropout_hidden = tf.layers.dropout(inputs=hidden,
        #                                   rate=0.2,
        #                                   training=training)

        logits = tf.layers.dense(inputs=dropout_flat, units=2)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:

            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predictions)}

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )

        # This will be None when predicting
        if labels is not None:
            onehot_labels = tf.one_hot(labels, 2, 1, 0)
            #labels = tf.reshape(labels, [-1, 1])

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        metric_ops = {
            "accuracy": tf.metrics.accuracy(labels, predictions['classes']),
            "auc": tf.metrics.auc(labels, predictions['classes'])
        }

        for metric_type, metric_op in metric_ops.items():
            # v[1] is the update op of the metrics object
            tf.summary.scalar(metric_type, metric_op[1])


        if mode == tf.estimator.ModeKeys.EVAL:
            # Add evaluation metrics (for EVAL mode)

            return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=metric_ops)


        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))

    def train(self):
        # setup train spec
        hook = tf.estimator.experimental.stop_if_no_increase_hook(
                self.classifier,
                'accuracy',
                300,
                min_steps=500,
                run_every_steps=100,
                run_every_secs=None
        )

        train_spec = tf.estimator.TrainSpec(
                input_fn=functools.partial(self.input_fn,
                                           self.config['datasets']['train'],
                                           shuffle_and_repeat=True),
                max_steps=self.config['num_epochs'],
                hooks=[hook]
        )

        # setup eval spec evaluating ever n seconds
        eval_spec = tf.estimator.EvalSpec(
                input_fn=functools.partial(self.input_fn,
                                           self.config['datasets']['eval']),
                steps=100,
                throttle_secs=60
        )

        # run train and evaluate
        tf.estimator.train_and_evaluate(self.classifier, train_spec, eval_spec)
        #self.classifier.evaluate(input_fn=self.eval_input_fn)

    def train_old(self):
        """Training process"""
        # Save a reference to the classifier to run predictions later
        self.classifier.train(input_fn=self.train_input_fn, steps=self.config['num_epochs'])

        #(x_test, y_test, y_length) = self.load_data_set(self.config['datasets']['eval'])

        eval_results = self.classifier.evaluate(input_fn=self.eval_input_fn, steps=100)

        #predictions = np.array([p['classes'][0] for p in self.classifier.predict(input_fn=self.eval_input_fn)])
        #print('Accuracy: {0:f}'.format(eval_results['accuracy']))
        #print('AUC: {0:f}'.format(eval_results['auc']))


        #predictions = list(self.classifier.predict(input_fn=self.eval_input_fn))
        #for p, l in zip(predictions, y_test):
        #    print(p['classes'], l)
