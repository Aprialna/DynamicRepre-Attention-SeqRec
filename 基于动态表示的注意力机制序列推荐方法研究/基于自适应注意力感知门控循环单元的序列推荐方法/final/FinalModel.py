import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

class Model(object):
    def __init__(self, batch_size, state_width, n_poi, n_user,is_training, learning_rate=0.0001, l2=1e-4):
        #learning_rate has been confirmed as 0.001 in SIN, don't change
        self.batch_size = batch_size
        self.state_width = state_width
        self.learning_rate = learning_rate
        self.n_poi = n_poi
        self.n_user = n_user
        self.l2 = l2
        self.input_data = tf.placeholder(tf.int32, [self.batch_size,None], name='input')
        self.pos = tf.placeholder(tf.int32, [self.batch_size], name='pos')
        self.neg = tf.placeholder(tf.int32, [self.batch_size], name='neg')

        self.l2_loss = tf.constant(0.0)
        item_embedding = tf.get_variable("item_embedding",[self.n_poi, self.state_width])

        # item_hidden_embedding  = tf.get_variable("item_hidden_embedding",[self.n_poi,self.state_width])
        # embeddingx
        history = tf.nn.embedding_lookup(item_embedding,self.input_data)
        pos = tf.nn.embedding_lookup(item_embedding,tf.expand_dims(self.pos,1))
        neg = tf.nn.embedding_lookup(item_embedding,tf.expand_dims(self.neg,1))

        with tf.name_scope('Input_Adaptive_Layer'):
            input_att_pos = tf.matmul(history, tf.transpose(pos,[0,2,1]))
            alpha_pos = tf.nn.softmax(input_att_pos, axis=0)
            input_attention_output_pos = tf.multiply(alpha_pos, history)

        input_attention_output_pos = tf.reshape(input_attention_output_pos,shape=[batch_size, -1, self.state_width])

        cell = GRUCell(self.state_width)

        # outputs:(batch_size,?,lstm_size)
        output_rnn_pos, final_state_pos = tf.nn.dynamic_rnn(cell=cell, inputs=input_attention_output_pos, dtype=tf.float32)

        with tf.name_scope('Hidden_Adaptive_Layer'):
            hidden_att_pos = tf.matmul(pos,tf.expand_dims(final_state_pos,2))
            hidden_alpha_pos = tf.nn.softmax(hidden_att_pos, axis=0)
            adaptive_user_pos = tf.multiply(hidden_alpha_pos,tf.expand_dims(final_state_pos,2))

        with tf.name_scope('BPR'):
            if not is_training:
                self.pred = tf.reduce_sum(tf.matmul(pos,adaptive_user_pos))
                return
            else:
                self.pred = -tf.reduce_sum(tf.matmul((pos-neg),adaptive_user_pos))

        self.losses = -tf.log(tf.sigmoid(self.pred))
        self.l2_loss += tf.nn.l2_loss(item_embedding)
        # self.l2_loss += tf.nn.l2_loss(softmax_b)
        self.l2_loss = self.l2 * self.l2_loss
        self.losses = self.l2_loss + self.losses
        with tf.name_scope('train'):
            self.cost = tf.div(tf.reduce_sum(self.losses, name='losses_sum'), batch_size, name='average_cost')

            tvars = tf.trainable_variables()
            '''grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), max_grad_norm)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.train_step = opt.apply_gradients(zip(grads, tvars))'''
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            # self.train_op = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.cost)

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
