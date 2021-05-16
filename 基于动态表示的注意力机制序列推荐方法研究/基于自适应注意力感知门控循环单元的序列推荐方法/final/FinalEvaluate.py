import random
from math import isnan

import tensorflow as tf
import numpy as np
from final.FinalModel import Model
# from final.HiddenModel import Model
N = 100
hidden_size = 128
max_grad_norm = 5
class Prediction:

    def __init__(self, path, train_seq, test_seq,popular_items, n_poi, n_user, K):
        self.path = path
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.popular_items = popular_items
        self.n_poi = n_poi
        self.n_user = n_user
        self.acc_K = K
        self.evaluation()
        self.compute_accuracy()

    def evaluation(self):
        model = Model(1, hidden_size, self.n_poi, self.n_user, is_training=False)
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                result_p = {}
                result_r = {}
                result_n = {}
                self.apks = list()
                self.score = 0
                self.rec = {}
                self.pre = {}
                self.ndcg = {}
                self.hit_num = {}

                self.record_num = 0
                for K in self.acc_K:
                    result_p[K] = 0
                    result_r[K] = 0
                    result_n[K] = 0
                    self.rec[K] = 0
                    self.pre[K] = 0
                    self.ndcg[K] = 0
                    self.hit_num[K] = 0

                allitems = list(range(self.n_poi)).copy()
                randomList = list(random.sample(allitems, N))
                final_pos = randomList+self.popular_items
                # final_pos = randomList
                for user_id in range(self.n_user):
                # for test_data in self.his_pos: # test_data为his_pos的key即pos,his_pos[test_data]为his_pos的value即C
                    hits = {}  # 记录每个用户的平均acc值
                    test_true = set(list(map(int,self.test_seq[user_id].strip(',').split(','))))
                    C = list(map(int,self.train_seq[user_id].strip(',').split(',')))
                    for K in self.acc_K:
                        hits[K] = 0
                    record_count = len(test_true)  # 记录一个用户的测试用例数
                    self.record_num += record_count

                    input = C # 用户训练集中的item，即C


                    pred = {}
                    for pos in final_pos:
                        feed = {model.input_data: [input],
                                     model.pos: [pos]}
                        score = sess.run(model.pred, feed_dict=feed)
                        pred[score] = pos
                    ranked_keys = sorted(pred.keys(),reverse= True)
                    ranked_values = []
                    for key in ranked_keys:
                        ranked_values.append(pred[key])

                    sorted_indexs = {}

                    for K in self.acc_K:
                        sorted_indexs[K] = ranked_values[:K]  # sort by values

                    # hit or not hit
                    for K in self.acc_K:
                        hits[K] += len(set(sorted_indexs[K]).intersection(test_true))
                        self.hit_num[K] += hits[K]

                    self.apks.append(self._compute_apk(test_true, ranked_values, k=np.inf))

                    for K in self.acc_K:
                        if record_count != 0:
                            result_r[K] = hits[K] / record_count
                            result_p[K] = hits[K] / K

                    for K in self.acc_K:
                        self.pre[K] += result_p[K]
                        self.rec[K] += result_r[K]

    def compute_accuracy(self):
        for K in self.acc_K:
            self.pre[K] /= self.n_user
            self.rec[K] /= self.n_user
            print(
                'average_precision@{}:\t{:.6f}\taverage_recall@{}:\t{:.6f}'
                .format(K, self.pre[K], K, self.rec[K]))
        mean_ap = np.mean(self.apks)
        print('average_map:{}'.format(mean_ap))

    def _compute_apk(self, targets, predictions, k):

        if len(predictions) > k:
            predictions = predictions[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predictions):
            if p in targets and p not in predictions[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not list(targets):
            return 0.0

        return score / min(len(targets), k)
