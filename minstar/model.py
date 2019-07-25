import numpy as np
import tensorflow as tf

from minstar.config import *

# Positional Encoding
def position_encoding(scaling=True, scope=None):
    # --------------------------- Output --------------------------- #
    # outputs : positional encoded matrix shape of (32, 20, 512) zero padded.

    with tf.variable_scope(scope):
        position_idx = tf.tile(tf.expand_dims(tf.range(FLAGS.sentence_maxlen),0), [FLAGS.batch_size,1]) # (32, 20)

        # PE_(pos, 2i) = sin(pos / 10000 ^ (2i / d_model))
        # PE_(pos, 2i+1) = cos(pos / 10000 ^ (2i / d_model))
        position_enc = np.array([[pos / (10000 ** (2*i / FLAGS.model_dim)) for i in range(FLAGS.model_dim)]
                                if pos != 0 else np.zeros(FLAGS.model_dim) for pos in range(FLAGS.sentence_maxlen)],
                                dtype=np.float32)

        # index 0 is all zero
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # sine functions to 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # cosine functions to 2i + 1

        # convert to tensor
        table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        outputs = tf.nn.embedding_lookup(table, position_idx)

        # output embedding scaling if needed
        if scaling:
            print ("position encoding scaling is executed")
            outputs *= (FLAGS.model_dim ** 0.5)

    return outputs

# Inputs and Outputs embedding lookup function
def embedding(inputs, input_vocab, padding=True, scaling=True, scope=None):
    # --------------------------- Input --------------------------- #
    # inputs : (batch_size, sentence max length) shape of input dataset
    # input_vocab : class, composed of token to index dictionary and reversed dictionary

    # --------------------------- Output --------------------------- #
    # outputs : embedding matrix shape of (32, 20, 512) zero padded.
    print ("token number :",len(input_vocab.token2idx))

    with tf.variable_scope(scope):
        table = tf.get_variable("word_embedding", shape=[len(input_vocab.token2idx), FLAGS.model_dim], \
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        if padding:
            table = tf.concat((tf.zeros(shape=[1, FLAGS.model_dim]), table[1:]), axis=0)

        # table : (123154, 512), inputs : (32, 20)
        outputs = tf.nn.embedding_lookup(table, inputs)

        if scaling:
            print ("embedding scaling is executed")
            outputs *= (FLAGS.model_dim ** 0.5)

    return outputs

# Layer Normalization
def layer_norm(inputs, scope=None):
    # --------------------------- Input --------------------------- #
    # inputs : multi-head attention outputs, shape of (batch_size, sentence max length, model dimension)

    # --------------------------- Output --------------------------- #
    # outputs : layer normalized with inputs, shape of (batch_size, sentence max length, model dimension)

    with tf.variable_scope(scope):
        mean, variance = tf.nn.moments(inputs, axes=2, keep_dims=True) # get mean and variance per batch.
        gamma = tf.Variable(tf.ones(inputs.get_shape()[2]))
        beta  = tf.Variable(tf.zeros(inputs.get_shape()[2]))
        normalized = (inputs - mean) / tf.sqrt(variance + 1e-5)
        outputs = gamma * normalized + beta # (32, 20, 512, dtype=float32)

    return outputs

# Multi-Head Attention
def multihead_attention(inputs, encoded_output=None, decoding=False, masking=False, dropout=True, scope=None):
    with tf.variable_scope(scope):
        # queries, keys, values come from the same place which is input

        if decoding:
            queries, keys, values = inputs, encoded_output, encoded_output
        else:
            queries, keys, values = inputs, inputs, inputs

        # linear transformation
        Q = tf.layers.dense(queries, FLAGS.model_dim, activation=tf.nn.relu, use_bias=True) # (32, 20, 512)
        K = tf.layers.dense(keys, FLAGS.model_dim, activation=tf.nn.relu, use_bias=True)    # (32, 20, 512)
        V = tf.layers.dense(values, FLAGS.model_dim, activation=tf.nn.relu, use_bias=True)  # (32, 20, 512)

        Q_concat = tf.concat(tf.split(Q, FLAGS.multi_head, axis=2), axis=0) # (8 * 32, 20, 64)
        K_concat = tf.concat(tf.split(K, FLAGS.multi_head, axis=2), axis=0) # (8 * 32, 20, 64)
        V_concat = tf.concat(tf.split(V, FLAGS.multi_head, axis=2), axis=0) # (8 * 32, 20, 64)

        # Multiplication
        K_transpose = tf.transpose(K_concat, perm=[0, 2, 1]) # (256, 64, 20)
        logits = tf.matmul(Q_concat, K_transpose)            # (256, 20, 20)

        # Scaling because of variance maintenance
        logits /= FLAGS.key_dim ** 0.5

        # Masking (optional. for decoding)
        if masking:
            diag_vals = tf.ones_like(logits[0,:,:]) # (20, 20)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [logits.get_shape()[0], 1, 1]) # (8 * 32, 20, 64)
            paddings = tf.ones_like(masks) * (-1.0e9)
            logits = tf.where(tf.equal(masks, 0), paddings, logits)

        # Softmax and multiply
        outputs = tf.nn.softmax(logits) # (256, 20, 20)

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=FLAGS.dropout ,training=dropout)

        # Context Vectore, denoted as Attention(Q, K, V)
        outputs = tf.matmul(outputs, V_concat) # (256, 20, 64)

        outputs = tf.concat(tf.split(outputs, FLAGS.multi_head, axis=0), axis=2) # (32, 20, 8 * 64)

        # Residual connection
        outputs += inputs # (32, 20, 512)

        # Normalize
        outputs = layer_norm(outputs, scope="layer_norm")

    return outputs

# Position-Wise Feed-Forward Networks
def position_ffn(inputs, scope=None):
    with tf.variable_scope(scope):
        hidden_layer = tf.layers.dense(inputs, FLAGS.inner_layer, activation=tf.nn.relu, use_bias=True) # (32, 20, 2048)
        output_layer = tf.layers.dense(hidden_layer, FLAGS.model_dim, activation=tf.nn.relu, use_bias=True) # (32, 20, 512)

        # residual connection
        output_layer += inputs # (32, 20, 512)

        # Normalize
        outputs = layer_norm(output_layer, scope="layer_norm") # (32, 20, 512)

    return outputs

# Encoder Stacked layers
def encoding_stack(inputs, num_stack=FLAGS.stack_layer):
    # --------------------------- Input --------------------------- #
    # inputs : (batch_size, sentence max length, model dimension)

    # --------------------------- Output --------------------------- #
    # inputs : output of stacked layer
    for idx in range(num_stack):
        with tf.variable_scope("encoding_stack_{}".format(idx)):
            hidden = multihead_attention(inputs, encoded_output=None, decoding=False, masking=False, \
                                        dropout=True, scope="self_att_enc")
            inputs = position_ffn(hidden, scope="enc_pffn") # (32, 20, 512)

    return inputs

# Decoder Stacked layers
def decoding_stack(inputs, enc_input, num_stack=FLAGS.stack_layer):
    # --------------------------- Input --------------------------- #
    # inputs : (batch_size, sentence max length, model dimension) of output source
    # enc_input : (batch_size, sentence max length, model dimension) of encoded output

    # --------------------------- Output --------------------------- #
    # inputs : output of stacked layer

    for idx in range(num_stack):
        with tf.variable_scope("decoding_stack_{}".format(idx)):
            hidden = multihead_attention(inputs, encoded_output=None, decoding=False, masking=True, \
                                        dropout=True, scope="mask_dec")
            enc_added = multihead_attention(hidden, encoded_output=enc_input, decoding=True, masking=False, \
                                            dropout=True, scope="self_att_dec")
            inputs = position_ffn(enc_added, scope="dec_pffn")

    return inputs

# total model graph
class model_graph():
    def __init__(self, source=None, target=None):
        # x, y : source _ (64, 40), target _ (64, 40)
        self.source = source
        self.target = target

        self.enc_inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.sentence_maxlen], name="enc_inputs")
        self.dec_inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.sentence_maxlen], name="dec_inputs")

        with tf.variable_scope("Encoding"):
            # input embedding lookup with table, source = de_vocab
            self.emb_outputs_enc = embedding(self.enc_inputs, input_vocab=self.source, padding=True, scaling=True, scope="enc_embed")
            #aaa = self.emb_outputs_enc.detach().numpy()

            # added positional encoding to embedding matrix
            self.pos_outputs_enc = position_encoding(scaling=True, scope="enc_pe")
            self.emb_outputs_enc += self.pos_outputs_enc

            # Stacked layer (Encoder)
            # multi-head attention, residual connection and Layer normalization
            # Feed Forward, residual connection and Layer normalization
            self.enc_outputs = encoding_stack(self.emb_outputs_enc, num_stack=FLAGS.stack_layer)

        with tf.variable_scope("Decoding"):
            # input embedding lookup with table, target = en_vocab
            self.emb_outputs_dec = embedding(self.dec_inputs, input_vocab=self.target, padding=True, scaling=True, scope="dec_embed")

            # added positional encoding to embedding matrix
            self.pos_outputs_dec = position_encoding(scaling=True, scope="dec_pe")
            self.emb_outputs_dec += self.pos_outputs_dec

            # Stacked layer (Decoded)
            # multi-head attention, residual connection and Layer normalization
            # Feed Forward, residual connection and Layer normalization
            self.dec_outputs = decoding_stack(self.emb_outputs_dec, self.enc_outputs, num_stack=FLAGS.stack_layer)

        # Linear Transformation
        self.logits = tf.layers.dense(self.dec_outputs, len(self.target.token2idx)) # (32, 20, 207203)
        self.pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32)

        # onehot encoding to use as label in loss function
        self.y_onehot = tf.one_hot(self.dec_inputs, depth=len(self.target.token2idx))
        print()

    def loss_fn(self):
        with tf.variable_scope("loss_function"):
            self.is_target = tf.to_float(tf.not_equal(self.dec_inputs, 0))
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_onehot)
            self.loss = tf.reduce_sum(self.cross_entropy * self.is_target) / (tf.reduce_sum(self.is_target) + 1e-8)
            # self.accuracy = tf.reduce_sum(tf.to_float(tf.equal(self.pred, self.dec_inputs)) * self.is_target) / (tf.reduce_sum(self.is_target))

        return self.is_target, self.loss

    def train_fn(self):
        with tf.variable_scope("train_function"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1, \
                                                    beta2=FLAGS.beta2, epsilon=FLAGS.adam_epsilon)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        return self.global_step, self.optimizer, self.train_op
