import os
import tensorflow as tf
import numpy as np
import functools
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.contrib import predictor

from ..data_loader import WordVector, TFDataReader, tokenize
from .. import LOGGER
from .utils import TrainHelper, FileHelper
from .tf_best_export import BestCheckpointsExporter


class TFMultiFeatClassifier:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']
        self.data_sets = {}
        self.embedding = None
        self.max_sequence_length = config['max_sequence_length']
        self.data_reader = TFDataReader(self.config)

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
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def load_embedding(self):
        if self.embedding is None:
            self.embedding = WordVector(self.config['embedding']['file'])

    def _inputs_to_features(self, inputs):
        ''' convert the text input to
            - token ids if unit is token
            - char ids if unit is char
        '''

        features = [
            [
                [self.embedding.get_index(token)
                 for token in tokenize(text)]
                for text in texts
            ]
            for texts in inputs
        ]

        data_length = [
            [
                min(len(feature), self.max_sequence_length[column_index])
                for column_index, feature in enumerate(feature_item)
            ]
            for feature_item in features
        ]
        data = [
            sequence.pad_sequences(
                feature_column,
                maxlen=self.max_sequence_length[column_index],
                truncating='post',
                padding='post',
                value=WordVector.PAD_ID).tolist()
            for column_index, feature_column in enumerate(zip(*features))
        ]

        return (data, data_length)

    def load_data_set(self, data_path):
        if data_path not in self.data_sets:
            inputs, labels = self.data_reader.get_data(data_path)
            (data, data_length) = self._inputs_to_features(inputs)
            self.data_sets[data_path] = (data, np.array(labels), data_length)
        return self.data_sets[data_path]

    @staticmethod
    def _data_parser(length, label, *inputs):
        features = {"len": length}
        for index, input in enumerate(inputs):
            features["input_" + str(index)] = input
        return features, label

    def input_fn(self, data_path, shuffle_and_repeat=False):
        LOGGER.info("load data from %s", data_path)
        (data, labels, data_length) = self.load_data_set(data_path)

        dataset = tf.data.Dataset.from_tensor_slices((data_length,
                                                      labels,
                                                      *data))
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

        to_merge = []
        # graph_selector = GraphSelector(self.config, self.embedding)
        for i in range(2):
            feature = features["input_" + str(i)]
            input_layer = tf.contrib.layers.embed_sequence(
                feature,
                self.embedding.vocab_size,
                self.embedding.vector_size,
                initializer=params['embedding_initializer'],
                trainable=False
            )

            dropout_emb = tf.layers.dropout(inputs=input_layer,
                                            rate=self.config['dropout_rate'],
                                            training=training)

            conv = tf.layers.conv1d(
                inputs=dropout_emb,
                filters=self.config['cnn']['filter_size'],
                kernel_size=self.config['cnn']['kernel_size'],
                padding='same',
                activation=tf.nn.relu)

            pool = tf.reduce_max(input_tensor=conv, axis=1)
            flat = tf.layers.dense(inputs=pool, units=256,
                                   activation=tf.nn.relu)
            to_merge.append(flat)

        merged = tf.concat(to_merge, 1)

        logits = tf.layers.dense(inputs=merged, units=2)

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
    def serving_input_receiver_fn(max_sequence_length):
        '''
        input shape:
           input_0: [batch_size, max_sequence_length[0] ]
           input_1: [batch_size, max_sequence_length[1] ]
           ......
           len: [batch_size, number_of_input ]
        '''
        features = {}
        receiver_tensors = {}

        seq_length = tf.placeholder(
                dtype=tf.int32,
                shape=[None, len(max_sequence_length)],
                name='seq_length'
        )
        features['len'] = seq_length
        for index, length in enumerate(max_sequence_length):
            input_name = 'input_' + str(index)
            input = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, length],
                    name=input_name
            )
            features[input_name] = input
            receiver_tensors[input_name] = input
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
                serving_input_receiver_fn=functools.partial(
                                              self.serving_input_receiver_fn,
                                              self.max_sequence_length
                                          ),
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
        self.model = predictor.from_saved_model(model_path)
        self._load_vocab()

    def _load_vocab(self):
        vocab, _ = WordVector.read_embeddings(self.config['embedding']['file'])
        self.vocab_to_ids = WordVector.create_vocab_index_dict(vocab)

    def process_with_saved_model(self, input):
        data = self._input_text_to_pad_id(input)
        result = self.model(data)
        probabilities = result['probabilities'][0]
        return probabilities.tolist()

    def _input_text_to_pad_id(self, texts):
        features = [[
                self.vocab_to_ids[token]
                if token in self.vocab_to_ids else WordVector.UNK_ID
                for token in tokenize(text)]
                for text in texts]
        data = [sequence.pad_sequences(
            [feature_column],
            maxlen=self.max_sequence_length[column_index],
            truncating='post',
            padding='post',
            value=WordVector.PAD_ID)
            for column_index, feature_column in enumerate(features)
        ]
        return {
                    'input_' + str(index): data[index]
                    for index in range(len(data))
                }
