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
from ..data_loader.data_reader import get_tf_data, tokenize
from .. import LOGGER
from .utils import TrainHelper
from .graph_selector import GraphSelector
from .tf_best_export import BestCheckpointsExporter

'''
TODO:
   - save all ckpt, but remove if not better, which make the evaluation only using the best models
   - predict using serving, and export the model
   https://guillaumegenthial.github.io/serving-tensorflow-estimator.html
'''
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
        if 'test' in self.config['datasets']:
            self.evaluate_on_tests()

    def evaluate_on_tests(self):
        for test_set_name in self.config['datasets']['test']:
            data_path = self.config['datasets']['test'][test_set_name]
            predicted_classes = self.predict_batch(data_path)
            _, lables, data_length = self.load_data_set(data_path)
            scores = TrainHelper.evaluate_score_on_class(predicted_classes, lables)
            TrainHelper.print_test_score(test_set_name, scores)
            cm = TrainHelper.evaluate_confusion_matrix_binary_class(predicted_classes, lables)
            LOGGER.info("Confusion matrix:")
            print(cm)

    def evaluate(self):
        print("to implement")

    def predict_batch(self, data_path):

        predicted_classes = [
                predict['classes']
                for predict in
                self.classifier.predict(
                        input_fn=functools.partial(self.input_fn, data_path)
                )
        ]
        return predicted_classes


    def load_embedding(self):
        self.embedding = WordVector(self.config['embedding']['file'])

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
    def _data_parser(input, length, label):
        features = {"input": input, "len": length}
        return features, label

    def input_fn(self, data_path, shuffle_and_repeat=False):
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

        self.model_dir = self.config['model_path']

        run_config = tf.estimator.RunConfig(save_checkpoints_steps=self.config['check_per_steps'],
                                            save_summary_steps=self.config['check_per_steps'],
                                            model_dir=self.model_dir,
                                            keep_checkpoint_max=5)

        self.classifier = tf.estimator.Estimator(
                model_fn=self.model_fn,
                config=run_config,
                params=params
        )

    def model_fn(self, features, labels, mode, params):
        training = mode == tf.estimator.ModeKeys.TRAIN

        graph_selector = GraphSelector(self.config, self.embedding)

        logits = graph_selector.add_graph(
            features,
            training,
            params['embedding_initializer'])

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:

            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predictions)
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )

        # This will be None when predicting
        if labels is not None:
            onehot_labels = tf.one_hot(labels, 2, 1, 0)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        metric_ops = {
            "accuracy": tf.compat.v1.metrics.accuracy(labels, predictions['classes']),
            "auc": tf.compat.v1.metrics.auc(labels, predictions['classes'])
        }

        for metric_type, metric_op in metric_ops.items():
            # v[1] is the update op of the metrics object
            tf.compat.v1.summary.scalar(metric_type, metric_op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=metric_ops)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer(self.config['learning_rate'])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))

    @staticmethod
    def serving_input_receiver_fn():
        """For the sake of the example, let's assume your input to the network will be a 28x28 grayscale image that you'll then preprocess as needed"""
        input_text = tf.placeholder(
                dtype=tf.int32,
                shape=[None, 512],
                name='input_text'
        )
        seq_length = tf.placeholder(
                dtype=tf.int32,
                shape=[None],
                name='seq_length'
        )

        features = {'input' : input_text, 'len': seq_length}
        receiver_tensors = {'input': input_text}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


    def train(self):

        hook = tf.estimator.experimental.stop_if_no_increase_hook(
                self.classifier,
                'accuracy',
                self.config['max_steps_without_increase'],
                min_steps=self.config['min_train_steps'],
                run_every_steps=self.config['check_per_steps'],
                run_every_secs=None
        )

        train_spec = tf.estimator.TrainSpec(
                input_fn=functools.partial(self.input_fn,
                                           self.config['datasets']['train'],
                                           shuffle_and_repeat=True),
                #max_steps=self.max_steps,
                hooks=[hook]
        )

        best_exporter = BestCheckpointsExporter(
                name="best_exporter",
                serving_input_receiver_fn=self.serving_input_receiver_fn,
                exports_to_keep=2
        )

        eval_spec = tf.estimator.EvalSpec(
                input_fn=functools.partial(self.input_fn,
                                           self.config['datasets']['eval']),
                exporters=best_exporter,
                throttle_secs=1
        )

        # train and evaluate
        tf.estimator.train_and_evaluate(self.classifier, train_spec, eval_spec)

    def separated_train_and_eval(self):
        """Training process"""
        # Save a reference to the classifier to run predictions later
        self.classifier.train(
                input_fn=functools.partial(self.input_fn,
                                          self.config['datasets']['train']),
        )
        eval_results = self.classifier.evaluate(
                input_fn=functools.partial(self.input_fn,
                                           self.config['datasets']['eval']),
                steps=self.config['check_per_steps']
        )
        print('Accuracy: {0:f}'.format(eval_results['accuracy']))
        print('AUC: {0:f}'.format(eval_results['auc']))

        predictions = np.array([p['classes'][0] for p in self.classifier.predict(input_fn=self.eval_input_fn)])
