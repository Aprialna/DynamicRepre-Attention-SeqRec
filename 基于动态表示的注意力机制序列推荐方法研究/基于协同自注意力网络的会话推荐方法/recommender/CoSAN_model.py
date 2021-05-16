import random

from component.modules import *
from recommender.TrainEvaluate import TrainEvaluate
import time

class CoSAN_model(TrainEvaluate):

    def __init__(self, dataModel, config):

        super(CoSAN_model, self).__init__(dataModel, config)
        self.train_users = dataModel.train_users

        self.train_sequences_input = dataModel.train_sequences_input
        self.train_sequences_user_input = dataModel.train_sequences_user_input
        self.train_sequences_user_weight = dataModel.train_sequences_user_weight
        self.train_sequences_target = dataModel.train_sequences_target
        self.user_pred_sequences = dataModel.user_pred_sequences
        self.user_pred_user_sequences = dataModel.user_pred_user_sequences
        self.user_pred_user_sequences_weight = dataModel.user_pred_user_sequences_weight

        "testSize 和 numUser 数量上比较一致"
        self.trainSize = len(self.train_users)
        "这个是一个epoch中有几个batch，而不是一个batch中有多少个number，每个epoch中同时对多个user进行训练"
        self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1
        self.familiar_user_num = dataModel.familiar_user_num

        self.numFactor = config['numFactor']
        self.input_length = config['input_length']
        self.dropout_rate = config['dropout_rate']
        self.keep_prob_user = config['keep_prob_user']
        self.num_units = config['cell_numbers']

        # placeholders
        self.u_id = tf.placeholder(tf.int32, [self.trainBatchSize, 1])
        self.input_seq = tf.placeholder(tf.int32, [self.trainBatchSize, (self.input_length-1)])
        self.input_user_seq = tf.placeholder(tf.int32, [self.trainBatchSize, (self.input_length-1) * self.familiar_user_num])
        self.input_user_weight = tf.placeholder(tf.float32, [self.trainBatchSize, (self.input_length-1) * self.familiar_user_num])
        self.target_seq_pos = tf.placeholder(tf.int32, [self.trainBatchSize, (self.input_length-1)])
        self.target_seq_neg = tf.placeholder(tf.int32, [self.trainBatchSize, (self.input_length-1)])

        self.u_id_test = tf.placeholder(tf.int32, [self.testBatchSize, 1])
        self.input_seq_test = tf.placeholder(tf.int32, [self.testBatchSize, (self.input_length-1)])
        self.input_user_seq_test = tf.placeholder(tf.int32, [self.testBatchSize, (self.input_length-1) * self.familiar_user_num])
        self.input_user_weight_test = tf.placeholder(tf.float32, [self.testBatchSize, (self.input_length-1) * self.familiar_user_num])
        self.pred_seq = tf.placeholder(tf.int32, [len(self.itemsInTestSet)]) # 候选集


        "dropout层的应用"
        self.dropout_keep_placeholder = tf.placeholder_with_default(1.0, shape=())
        # user/item embedding
        "numFactor为embedding的size"
        with tf.variable_scope("Embedding"):
            self.userBias_gru = tf.Variable(tf.random_normal([self.numFactor], 0, 0.1))

        self.C_userEmbedding = tf.Variable(tf.random_normal([self.numUser, self.numFactor], 0, 1/tf.sqrt(float(self.numFactor))))
        self.C_itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 1/tf.sqrt(float(self.numFactor))))
        self.position = tf.Variable(tf.random_normal([self.input_length, self.numFactor], 0, 1/tf.sqrt(float(self.numFactor))))

        self.userC_weight = tf.Variable(tf.constant(0.5, shape=[1, self.numFactor]))


    def buildModel(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:

            userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.u_id),
                                       [-1, self.numFactor])
            user_embedding_drop = tf.nn.dropout(userEmbedding, self.keep_prob_user)

            userEmbedding_test = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.u_id_test),
                                            [-1, self.numFactor])

            C_familiar_userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.input_user_seq),
                                                  [-1, self.numFactor])
            C_familiar_user_embedding_drop = tf.nn.dropout(C_familiar_userEmbedding, self.keep_prob_user)

            C_familiar_user_embedding_test = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.input_user_seq_test),
                                                        [-1, self.numFactor])

            "memory: trainSize * input_seq * numfactor"
            C_memory = tf.nn.embedding_lookup(self.C_itemEmbedding, self.input_seq)

            C_memory_test = tf.nn.embedding_lookup(self.C_itemEmbedding, self.input_seq_test)

            target_item_test = tf.nn.embedding_lookup(self.C_itemEmbedding, self.pred_seq)


            rating_loss = self.train_pred(user_embedding_drop, C_familiar_user_embedding_drop, C_memory,
                                            self.target_seq_pos, self.target_seq_neg, self.input_user_weight)
            rating_loss = tf.reduce_mean(rating_loss)
            self.cost = rating_loss
            self.r_pred = self.test_pred(userEmbedding_test, C_familiar_user_embedding_test,
                                                 C_memory_test, target_item_test, self.input_user_weight_test)


    def train_pred(self, user_embedding_drop, C_familiar_user_embedding_drop, C_memory, target_item_ids,
                     negative_item_ids,weight):

        C_user_memory_out = tf.reshape(self.dynamic_item_block_user(C_familiar_user_embedding_drop,weight),
                                       [-1, (self.input_length-1), self.numFactor])

        C_item_pre_embedding = tf.add(C_memory, tf.multiply(tf.clip_by_value(self.userC_weight, 0.1, 1.0),
                                                            C_user_memory_out))
        item_embed_input = tf.nn.dropout(tf.reshape(C_item_pre_embedding, [-1, (self.input_length-1), self.numFactor]),
                                         self.keep_prob_user)

        # ================================SAN的表示=================================================
        # Build blocks，假设num-block=2
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        position = tf.nn.embedding_lookup(self.position,tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]),0), [tf.shape(self.input_seq)[0], 1]))
        item_embed_input += position # 将输入与position embedding元素相加

        # Dropout
        item_embed_input = tf.layers.dropout(item_embed_input,rate= self.dropout_rate)
        item_embed_input *= mask
        for i in range(2):
            with tf.variable_scope("num_blocks_%d" % i,reuse=tf.AUTO_REUSE):
                # Self-attention
                self.seq = multihead_attention(queries=normalize(item_embed_input),
                                               keys=item_embed_input,
                                               num_units=self.num_units,
                                               num_heads=1,
                                               dropout_rate= self.dropout_rate,
                                               causality=True)

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[self.num_units, self.num_units],
                                       dropout_rate=self.dropout_rate)
                self.seq *= mask

        san_outputs = normalize(self.seq)

        # ==================================END===================================================

        user_embedding_drop = tf.tanh(user_embedding_drop + self.userBias_gru)
        # SAN, multiply & add
        user_embedding_drop = tf.add(san_outputs, tf.expand_dims(user_embedding_drop, axis=1))

        # user_embedding_drop = tf.reshape(user_embedding_drop, [-1, self.numK, self.numFactor])
        target_item_ids = tf.reshape(target_item_ids, [self.trainBatchSize * (self.input_length - 1)])
        negative_item_ids = tf.reshape(negative_item_ids, [self.trainBatchSize * (self.input_length - 1)])
        user_embedding_drop = tf.reshape(user_embedding_drop,
                                         [self.trainBatchSize * (self.input_length - 1), self.numFactor])
        pos_embedding = tf.nn.embedding_lookup(self.C_itemEmbedding, target_item_ids)
        neg_embedding = tf.nn.embedding_lookup(self.C_itemEmbedding, negative_item_ids)


        element_pos = tf.reduce_sum(user_embedding_drop * pos_embedding, -1)  # [batch,len]
        element_neg = tf.reduce_sum(user_embedding_drop * neg_embedding, -1)


        log_loss = tf.reduce_mean(
            - tf.log(tf.sigmoid(element_pos) + 1e-24) -
            tf.log(1 - tf.sigmoid(element_neg) + 1e-24))
        log_intention = tf.reshape(log_loss, [-1, 1])
        return log_intention

    def test_pred(self, user_embedding_test, C_familiar_user_embedding_test, C_memory_test, target_item_test,weight):

        C_user_memory_out = self.dynamic_item_block_user(C_familiar_user_embedding_test,weight)

        C_item_pre_embedding = tf.add(C_memory_test, tf.multiply(tf.clip_by_value(self.userC_weight, 0.1, 1.0),
                                                                 C_user_memory_out))
        item_embed_input = tf.reshape(C_item_pre_embedding, [-1, (self.input_length-1), self.numFactor])

        # Build blocks，假设num-block=2
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq_test, 0)), -1)
        position = tf.nn.embedding_lookup(self.position,
                                          tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq_test)[1]), 0),
                                                  [tf.shape(self.input_seq_test)[0], 1]))
        item_embed_input += position  # 将输入与position embedding元素相加
        item_embed_input *= mask
        for i in range(2):
            with tf.variable_scope("num_blocks_%d" % i,reuse=tf.AUTO_REUSE):
                # Self-attention
                self.seq = multihead_attention(queries=normalize(item_embed_input),
                                               keys=item_embed_input,
                                               num_units=self.num_units,
                                               num_heads=1,
                                               is_training=False,
                                               causality=True)

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[self.num_units, self.num_units],
                                       is_training=False)
                self.seq *= mask

        san_outputs = normalize(self.seq)

        user_embedding_test = tf.tanh(user_embedding_test + self.userBias_gru)
        user_embedding_test = tf.add(san_outputs, tf.expand_dims(user_embedding_test, axis=1))
        target_item_test = tf.expand_dims(target_item_test,axis=0)
        self.test_logits = tf.matmul(user_embedding_test, tf.transpose(target_item_test, [0, 2, 1]))
        self.test_logits = tf.reshape(self.test_logits, [self.testBatchSize, self.input_length - 1, len(self.itemsInTestSet)])
        self.test_logits = self.test_logits[:, -1, :]
        return self.test_logits

    def dynamic_item_block_user(self,user_memory_embedding,weight):
        user_memory_embedding = tf.reshape(user_memory_embedding,
                                           [-1, (self.input_length-1) * self.familiar_user_num, self.numFactor])
        out =tf.reduce_mean(tf.multiply(
            tf.reshape(user_memory_embedding, [-1, (self.input_length-1), self.familiar_user_num, self.numFactor]),
            tf.expand_dims(tf.reshape(weight,[-1,(self.input_length-1),self.familiar_user_num]),axis=-1)), axis=2)
        "return shape: (train_batch, input_size, numFactor)"
        return out


    def trainEachBatch(self, epochId, batchId):
        totalLoss = 0
        start = time.time()
        feed_dict = self.getTrainData(batchId)

        self.optimizer.run(feed_dict=feed_dict)
        loss = self.sess.run(self.cost, feed_dict=feed_dict)

        totalLoss += loss
        end = time.time()
        if epochId % 2 == 0 and batchId == 0:
            self.logger.info(
                "batchId: %d epoch %d/%d   batch_loss: %.4f   time of a batch: %.4f" % (
                    batchId, epochId, self.maxIter, totalLoss, (end - start)))

            self.evaluate(epochId, batchId)
        return totalLoss

    def getTrainData(self, batchId):
        neg_seq_batch = []

        start_idx = batchId * self.trainBatchSize
        end_idx = start_idx + self.trainBatchSize

        if end_idx > self.trainSize:
            end_idx = self.trainSize
            start_idx = end_idx - self.trainBatchSize

        if end_idx == start_idx:
            start_idx = 0
            end_idx = start_idx + self.trainBatchSize
        "从一个连续的用户列表如[1,1,1,1,1,1,2,2,3....]中选取，存在很多重复的userIdx"
        user_batch = self.train_users[start_idx:end_idx]
        input_seq_batch = self.train_sequences_input[start_idx:end_idx]
        input_user_seq_batch = self.train_sequences_user_input[start_idx:end_idx]
        input_user_seq_weight_batch = self.train_sequences_user_weight[start_idx:end_idx]

        pos_seq_batch = self.train_sequences_target[start_idx:end_idx]

        for Idx in range(len(user_batch)):
            neg_items = []
            positiveItems = pos_seq_batch[Idx]
            for i in range(self.input_length-1):
                negativeItemIdx = random.randint(0, self.numItem - 1)
                while negativeItemIdx in positiveItems:
                    negativeItemIdx = random.randint(0, self.numItem - 1)
                neg_items.append(negativeItemIdx)
            neg_seq_batch.append(neg_items)

        user_batch = np.array(user_batch).reshape((end_idx - start_idx, 1))
        input_seq_batch = np.array(input_seq_batch)
        input_user_seq_batch = np.array(input_user_seq_batch).reshape((-1, (self.input_length-1) * self.familiar_user_num))
        input_user_seq_weight_batch = np.array(input_user_seq_weight_batch).reshape((-1, (self.input_length-1) * self.familiar_user_num))
        pos_seq_batch = np.array(pos_seq_batch)
        neg_seq_batch = np.array(neg_seq_batch)

        feed_dict = {
            self.u_id: user_batch,
            self.input_seq: input_seq_batch,
            self.input_user_seq: input_user_seq_batch,
            self.input_user_weight:input_user_seq_weight_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_rate
        }

        return feed_dict

    "test时使用，为每个user创建一个包含正样本item和负样本item的list"
    def getPredList_ByUserIdxList(self, user_idices):
        end0 = time.time()
        input_seq = []
        input_user_seq = []
        input_user_seq_weight = []
        target_seq = list(x+1 for x in self.itemsInTestSet)

        for userIdx in user_idices:
            "input_seq中存放的是用来预测每个user最新的item的前几个item，作为input"
            input_seq.append(self.user_pred_sequences[userIdx])
            input_user_seq.append(self.user_pred_user_sequences[userIdx])
            input_user_seq_weight.append(self.user_pred_user_sequences_weight[userIdx])

        batch_u = np.array(user_idices).reshape((-1, 1))
        input_seq = np.array(input_seq)
        input_user_seq = np.array(input_user_seq).reshape((-1, (self.input_length-1) * self.familiar_user_num))
        input_user_seq_weight = np.array(input_user_seq_weight).reshape((-1, (self.input_length-1) * self.familiar_user_num))
        target_seq = np.array(target_seq)

        end1 = time.time()

        predList = self.sess.run(self.r_pred, feed_dict={
            self.u_id_test: batch_u,
            self.input_seq_test: input_seq, # test input
            self.input_user_seq_test: input_user_seq, # test neighbor
            self.input_user_weight_test: input_user_seq_weight,
            self.pred_seq: target_seq, # candidate
        })
        end2 = time.time()

        output_lists = []

        for score in predList: # score对应了每个用户的预测得分
            score = list(score)
            score_to_item = {k:v for k,v in zip(score,list(target_seq))}
            score.sort(reverse = True) # 对得分从大到小排序并取topN个最高得分
            topN_score = score[:self.topN]
            topN_items = [score_to_item[k] for k in topN_score]
            output_lists.append(topN_items)

        end3 = time.time()

        return output_lists, end1 - end0, end2 - end1, end3 - end2

