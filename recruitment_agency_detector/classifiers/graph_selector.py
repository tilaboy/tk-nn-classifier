import tensorflow as tf

class GraphSelector:
    def __init__(self, config, embedding):
        self.config = config
        self.embedding = embedding

    def add_graph(self, input, training_mode, embedding_initializer):
        if self.config['model_type'] == 'tf_cnn_simple':
            return self._cnn_simple(input, training_mode, embedding_initializer)
        elif self.config['model_type'] == 'tf_cnn_multi':
            return self._cnn_multi_layer(input, training_mode, embedding_initializer)
        elif self.config['model_type'] == 'tf_lstm_simple':
            return self._lstm_simple(input, training_mode, embedding_initializer)
        elif self.config['model_type'] == 'tf_lstm_multi':
            return self._lstm_multi_layer(input, training_mode, embedding_initializer)


    def _cnn_simple(self, input, training_mode, embedding_initializer):

        input_layer = tf.contrib.layers.embed_sequence(
            input['x'],
            self.embedding.vocab_size,
            self.embedding.vector_size,
            initializer=embedding_initializer,
            trainable=False
        )

        dropout_emb = tf.layers.dropout(inputs=input_layer,
                                        rate=self.config['dropout_rate'],
                                        training=training_mode)

        conv = tf.layers.conv1d(
            inputs=dropout_emb,
            filters=self.config['cnn']['filter_size'],
            kernel_size=self.config['cnn']['kernel_size'],
            padding='same',
            activation=tf.nn.relu)

        pool = tf.reduce_max(input_tensor=conv, axis=1)
        hidden = tf.layers.dense(inputs=pool, units=256, activation=tf.nn.relu)
        dropout_hidden = tf.layers.dropout(inputs=hidden,
                                           rate=self.config['dropout_rate'],
                                           training=training_mode)
        logits = tf.layers.dense(inputs=dropout_hidden, units=2)
        return logits

    def _cnn_multi_layer(self, input, training_mode, embedding_initializer):
        input_layer = tf.contrib.layers.embed_sequence(
            input['x'],
            self.embedding.vocab_size,
            self.embedding.vector_size,
            initializer=embedding_initializer,
            trainable=False
        )

        next_input = tf.layers.dropout(inputs=input_layer,
                                        rate=self.config['dropout_rate'],
                                        training=training_mode)

        for i in range(self.config['cnn']['nr_layers']):
            conv = tf.layers.conv1d(
                inputs=next_input,
                filters=self.config['cnn']['filter_size'],
                kernel_size=self.config['cnn']['kernel_size'],
                padding='same',
                activation=tf.nn.relu)

            next_input = tf.layers.max_pooling1d(
                inputs=conv,
                pool_size=2,
                strides=2,
                padding='same')

        flat = tf.contrib.layers.flatten(next_input)
        dropout_flat = tf.layers.dropout(inputs=flat,
                                         rate=self.config['dropout_rate'],
                                         training=training_mode)

        logits = tf.layers.dense(inputs=dropout_flat, units=2)
        return logits

    def _lstm_simple(self, input, training_mode, embedding_initializer):
        input_layer = tf.contrib.layers.embed_sequence(
            input['x'],
            self.embedding.vocab_size,
            self.embedding.vector_size,
            initializer=embedding_initializer,
            trainable=False
        )
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.config['lstm']['hidden_size'])
        _, final_state = tf.nn.dynamic_rnn(
            cell, input_layer, sequence_length=input['len'], dtype=tf.float32)

        outputs = final_state.h
        logits = tf.layers.dense(inputs=outputs, units=2)
        return logits


    def _lstm_multi_layer(self, input, training_mode, embedding_initializer):
        input_layer = tf.contrib.layers.embed_sequence(
            input['x'],
            self.embedding.vocab_size,
            self.embedding.vector_size,
            initializer=embedding_initializer,
            trainable=False
        )
        rnn_layers = [
            tf.compat.v1.nn.rnn_cell.LSTMCell(self.config['lstm']['hidden_size'])
            for _ in range(self.config['lstm']['nr_layers'])
        ]

        multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)
        outputs, final_state = tf.compat.v1.nn.dynamic_rnn(
            cell=multi_rnn_cell,
            inputs=input_layer,
            sequence_length=input['len'],
            dtype=tf.float32
        )

        #final_outputs = tf.transpose(outputs, [1,0,2])[-1]
        final_outputs = final_state.h
        logits = tf.layers.dense(inputs=final_outputs, units=2)
        return logits
