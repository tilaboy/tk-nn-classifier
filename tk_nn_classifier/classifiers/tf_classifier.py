import os
import tensorflow as tf
import numpy as np
import pickle
import functools
from tensorflow.python.keras.preprocessing import sequence

from ..data_loader import WordVector, download_tk_embedding
from ..data_loader import TFDataReader, tokenize
from .. import LOGGER
from .utils import TrainHelper, FileHelper
from .graph_selector import GraphSelector
from .tf_best_export import BestCheckpointsExporter


class TFClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']
        self.max_sequence_length = config['max_sequence_length']
        self.data_sets = {}
        self.embedding = None
        self.data_reader = TFDataReader(self.config)
        os.makedirs(self.config['model_path'], exist_ok=True)

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
            _, labels, data_length = self.load_data_set(data_path)
            TrainHelper.print_test_result(predicted_classes, labels)

    def predict_batch(self, data_path):
        predicted_classes = [
                predict['classes']
                for predict in
                self.classifier.predict(
                        input_fn=functools.partial(self.input_fn, data_path)
                )
        ]
        return predicted_classes

    def _prepare_single_input(self, text):
        data_id = [
                self.embedding.get_index(token)
                for token in tokenize(text)
                ]
        data_length = min(len(data_id), self.max_sequence_length)
        data = sequence.pad_sequences([data_id],
                                      maxlen=self.max_sequence_length,
                                      truncating='post',
                                      padding='post',
                                      value=WordVector.PAD_ID)
        dataset = tf.data.Dataset.from_tensor_slices((data,
                                                      [data_length],
                                                      [0]))
        dataset = dataset.map(self._data_parser)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator.get_next()

    def load_embedding(self):
        target_file = self.config['embedding']['filepath']
        download_tk_embedding(self.config['language'], target_file)
        if self.embedding is None:
            self.embedding = WordVector(target_file)
            self._save_vocab_file()

    def _save_vocab_file(self):
        if self.embedding is not None:
            vocab_filename = os.path.join(self.config['model_path'], 'vocab.p')
            LOGGER.info('write vocab file to %s' % vocab_filename)
            with open(vocab_filename, 'wb') as handle:
                pickle.dump(self.embedding.vocab_to_index, handle)

    def load_data_set(self, data_path):
        if data_path not in self.data_sets:
            texts, labels = self.data_reader.get_data(data_path)

            data_ids = [[
                    self.embedding.get_index(token)
                    for token in tokenize(text)]
                    for text in texts]
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

        dataset = tf.data.Dataset.from_tensor_slices((data,
                                                      data_length,
                                                      labels))
        if shuffle_and_repeat:
            dataset = dataset.shuffle(buffer_size=len(data))
            dataset = dataset.repeat(self.config['num_epochs'])

        dataset = dataset.batch(self.config['batch_size'])
        dataset = dataset.map(self._data_parser)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def build_graph(self):
        def embedding_initializer(shape=None,
                                  dtype=tf.float32,
                                  partition_info=None):
            assert dtype is tf.float32
            return self.embedding.vectors

        # params = {'embedding_initializer':
        #           tf.random_uniform_initializer(-1.0, 1.0)}
        params = {'embedding_initializer': embedding_initializer}

        self.model_dir = self.config['model_path']

        run_config = tf.estimator.RunConfig(
            save_checkpoints_steps=self.config['check_per_steps'],
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
            onehot_labels = tf.one_hot(labels, 2, 1.0, 0.0)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                               logits=logits)

        metric_ops = {
            "accuracy": tf.compat.v1.metrics.accuracy(labels,
                                                      predictions['classes']),
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
            optimizer = tf.compat.v1.train.AdamOptimizer(
                self.config['learning_rate'])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.compat.v1.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)

        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))

    @staticmethod
    def serving_input_receiver_fn():
        '''serving input, to work with tensorflow estimator command tools like:
        saved_model_cli, also for prediction input
        '''
        input_text = tf.placeholder(
                dtype=tf.int32,
                shape=[None, 1024],
                name='input_text'
        )
        seq_length = tf.placeholder(
                dtype=tf.int32,
                shape=[None],
                name='seq_length'
        )

        features = {'input': input_text, 'len': seq_length}
        receiver_tensors = {'input': input_text}
        return tf.estimator.export.ServingInputReceiver(features,
                                                        receiver_tensors)

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
                # max_steps=self.max_steps,
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
        LOGGER.info("start model training %s", self.config['model_path'])
        tf.estimator.train_and_evaluate(self.classifier, train_spec, eval_spec)

    def predict_on_text(self, text):
        return self.classifier.predict(
            input_fn=functools.partial(self._prepare_single_input, text))

    def load_saved_model(self, model_path=None):
        if model_path is None:
            model_path = FileHelper.last_modified_folder(
                    os.path.join(
                            self.config['model_path'],
                            'export',
                            'best_exporter'
                            )
                    )
        LOGGER.info("loading model from %s", model_path)
        self.model = tf.contrib.predictor.from_saved_model(model_path)
        self._load_vocab()

    def _load_vocab(self):
        vocab, _ = WordVector.read_embeddings(self.config['embedding']['filepath'])
        self.vocab_to_ids = WordVector.create_vocab_index_dict(vocab)

    def process_with_saved_model(self, input):
        data = self._input_text_to_pad_id(input)
        result = self.model(data)
        probabilities = result['probabilities'][0]
        return probabilities.tolist()

    # todo:
    # the padding is probably not needed in the predicting mode
    # if needed, should use the text length as max_sequence_length
    def _input_text_to_pad_id(self, text):
        data_id = [
                self.vocab_to_ids[token]
                if token in self.vocab_to_ids else WordVector.UNK_ID
                for token in tokenize(text)
                ]
        data = sequence.pad_sequences([data_id],
                                      maxlen=self.max_sequence_length,
                                      truncating='post',
                                      padding='post',
                                      value=WordVector.PAD_ID)
        return {'input': data}
