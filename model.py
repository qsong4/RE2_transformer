import tensorflow as tf

from data_load import load_vocab, loadGloVe
from modules import get_token_embeddings, ff, multihead_attention, ln, positional_encoding_bert
import math

class FI:
    """
    xs: tuple of
        x: int32 tensor. (句子长度，)
        x_seqlens. int32 tensor. (句子)
    """

    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token, self.hp.vocab_size = load_vocab(hp.vocab)
        self.embd = None
        if self.hp.preembedding:
            self.embd = loadGloVe(self.hp.vec_path)
        self.embeddings = get_token_embeddings(self.embd, self.hp.vocab_size, self.hp.d_model, zero_pad=False)
        self.x = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_x")
        self.y = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_y")
        self.x_len = tf.placeholder(tf.int32, [None])
        self.y_len = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None, self.hp.num_class], name="truth")
        self.is_training = tf.placeholder(tf.bool,shape=None, name="is_training")


        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.global_step = self._globalStep_op()
        self.train = self._training_op()

    def create_feed_dict(self, x, y, x_len, y_len, truth, is_training):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.x_len: x_len,
            self.y_len: y_len,
            self.truth: truth,
            self.is_training: is_training,
        }

        return feed_dict

    def create_feed_dict_infer(self, x, y, x_len, y_len):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.x_len: x_len,
            self.y_len: y_len,
        }

        return feed_dict

    def representation(self, xs, ys):

        x = xs
        y = ys

        mask_a = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
        mask_b = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)
        mask_a = tf.expand_dims(mask_a, axis=-1)
        mask_b = tf.expand_dims(mask_b, axis=-1)

        # embedding
        encx = tf.nn.embedding_lookup(self.embeddings, x)  # (N, T1, d_model)
        encx *= self.hp.d_model ** 0.5  # scale

        #encx += positional_encoding(encx, self.hp.maxlen)
        encx += positional_encoding_bert(encx, self.hp.maxlen)
        encx = tf.layers.dropout(encx, self.hp.dropout_rate, training=self.is_training)

        ency = tf.nn.embedding_lookup(self.embeddings, y)  # (N, T1, d_model)
        ency *= self.hp.d_model ** 0.5  # scale

        #ency += positional_encoding(ency, self.hp.maxlen)
        ency += positional_encoding_bert(ency, self.hp.maxlen)
        ency = tf.layers.dropout(ency, self.hp.dropout_rate, training=self.is_training)
        # add ln
        encx = ln(encx)
        ency = ln(ency)

        emb_encx = encx
        emb_ency = ency

        for i in range(self.hp.blocks):
            with tf.variable_scope("blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                if i > 0:
                    encx = self.connection(encx, emb_encx, i)
                    ency = self.connection(ency, emb_ency, i)
                    emb_encx, emb_ency = encx, ency
                print(encx.shape)
                x_enc = self.encoder_blocks(encx)
                y_enc = self.encoder_blocks(ency)
                encx = tf.concat([encx, x_enc], axis=-1)
                ency = tf.concat([ency, y_enc], axis=-1)

                align_a, align_b = self.calculate_att(encx, ency, scope="alignment")

                encx = self.fusion(encx, align_a)
                ency = self.fusion(ency, align_b)
        encx = self.pooling(encx, mask_a)
        ency = self.pooling(ency, mask_b)

        return encx, ency

    def connection(self, x, res, i):
        if i == 1:
            x = tf.concat([res, x], axis=-1)  # res is embedding
        elif i > 1:
            hidden_size = int(x.shape[-1])
            x = (res[:, :, -hidden_size:] + x) * math.sqrt(0.5)
            x = tf.concat([res[:, :, :-hidden_size], x], axis=-1)  # former half of res is embedding
        return x

    def fusion(self, x, align):
        with tf.variable_scope('align', reuse=tf.AUTO_REUSE):
            x = tf.concat([
                tf.layers.dense(tf.concat([x, align], axis=-1), self.hp.d_model, activation=tf.nn.relu, name='orig'),
                tf.layers.dense(tf.concat([x, x - align], axis=-1), self.hp.d_model, activation=tf.nn.relu, name='sub'),
                tf.layers.dense(tf.concat([x, x * align], axis=-1), self.hp.d_model, activation=tf.nn.relu, name='mul'),
            ], axis=-1)
            x = tf.layers.dropout(x, self.hp.dropout_rate, training=self.is_training)
            x = tf.layers.dense(x, self.hp.d_model, activation=tf.nn.relu, name="proj")
            return x

    def pooling(self, x, mask):
        return tf.reduce_max(mask * x + (1. - mask) * tf.float32.min, axis=1)

    def encoder_blocks(self, a_repre, reuse=tf.AUTO_REUSE):
        for i in range(self.hp.num_transformer):
            with tf.variable_scope("num_trans_blocks_{}".format(i), reuse=reuse):
                # self-attention
                a_repre = multihead_attention(queries=a_repre,
                                           keys=a_repre,
                                           values=a_repre,
                                           num_heads=self.hp.num_heads,
                                           dropout_rate=self.hp.dropout_rate,
                                           training=self.is_training,
                                           causality=False)
                # feed forward
                #a_repre = ff(a_repre, num_units=[self.hp.d_ff, self.hp.d_model])
                a_repre = ff(a_repre, num_units=[self.hp.d_ff, a_repre.shape.as_list()[-1]])
        return a_repre

    def _attention(self, a, b, t, dropout_keep_prob):
        with tf.variable_scope('proj_atten'):
            a = tf.layers.dense(tf.layers.dropout(a, dropout_keep_prob, training=self.is_training),
                      self.hp.d_model, activation=tf.nn.relu)
            b = tf.layers.dense(tf.layers.dropout(b, dropout_keep_prob, training=self.is_training),
                      self.hp.d_model, activation=tf.nn.relu)
        return tf.matmul(a, b, transpose_b=True) * t

    def calculate_att(self, a, b, scope='alignment'):
        with tf.variable_scope(scope):
            mask_a = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
            mask_b = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)
            mask_a = tf.expand_dims(mask_a , axis=-1)
            mask_b = tf.expand_dims(mask_b , axis=-1)
            temperature = tf.get_variable('temperature', shape=(), dtype=tf.float32, trainable=True,
                                          initializer=tf.constant_initializer(math.sqrt(1 / self.hp.d_model)))
            attention = self._attention(a, b, temperature, self.hp.dropout_rate)
            attention_mask = tf.matmul(mask_a, mask_b, transpose_b=True)
            attention = attention_mask * attention + (1 - attention_mask) * tf.float32.min
            attention_a = tf.nn.softmax(attention, axis=1)
            attention_b = tf.nn.softmax(attention, axis=2)
            attention_a = tf.identity(attention_a, name='attention_a')
            attention_b = tf.identity(attention_b, name='attention_b')

            feature_b = tf.matmul(attention_a, a, transpose_a=True)
            feature_a = tf.matmul(attention_b, b)
            return feature_a, feature_b

    def fc_2l(self, inputs, num_units, scope="fc_2l"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])

        return outputs

    def _project_op(self, inputx):
        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
            inputx = tf.layers.dense(inputx, self.hp.d_model,
                                     activation=tf.nn.relu,
                                     name='fnn',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            return inputx



    def fc(self, inpt, match_dim, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fc", reuse=reuse):
            w = tf.get_variable("w", [match_dim, self.hp.num_class], dtype=tf.float32)
            b = tf.get_variable("b", [self.hp.num_class], dtype=tf.float32)
            logits = tf.matmul(inpt, w) + b
        # prob = tf.nn.softmax(logits)

        # gold_matrix = tf.one_hot(labels, self.hp.num_class, dtype=tf.float32)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return logits

    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.y, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    # 将transformer的每层作为一个channel输入CNN
    def cnn_agg(self, match_channels):
        # Create a convolution + maxpool layer for each filter size
        filter_sizes = list(map(int, self.hp.filter_sizes.split(",")))
        embedding_size = match_channels.shape.as_list()[2]
        sequence_length = match_channels.shape.as_list()[1]
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 6, self.hp.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.hp.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    match_channels,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.hp.num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.hp.dropout_rate)

        return h_drop

    def _logits_op(self):
        # representation
        x_repre, y_repre = self.representation(self.x, self.y)  # (layers, batchsize, maxlen, d_model)

        x = tf.concat([x_repre, y_repre, x_repre * y_repre, x_repre - y_repre], axis=-1)

        #logits = self.fc(x, match_dim=agg_res.shape.as_list()[-1])
        logits = self.fc_2l(x, num_units=[self.hp.d_model, self.hp.num_class], scope="fc_2l")
        return logits

    def _loss_op(self, l2_lambda=0.0001):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.truth))
        weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel') in v.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
        loss += l2_loss
        #添加aeloss
        return loss

    def _acc_op(self):
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.truth, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def _globalStep_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _training_op(self):
        # train scheme
        # global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        #optimizer = tf.train.GradientDescentOptimizer(self.hp.lr)
        # optimizer = tf.train.AdadeltaOptimizer(self.hp.lr)

        '''
        if self.hp.lambda_l2>0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            loss = loss + self.hp.lambda_l2 * l2_loss
        '''

        # grads = self.compute_gradients(loss, tvars)
        # grads, _ = tf.clip_by_global_norm(grads, 10.0)
        # train_op = optimizer.minimize(loss, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_op = optimizer.minimize(loss)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        return train_op


