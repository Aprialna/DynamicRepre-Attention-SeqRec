import numpy as np
import tensorflow as tf
import time
import eval.Metrics
import random

class TrainEvaluate:

    def __init__(self, dataModel, config):

        self.config = config
        tf.set_random_seed(config['random_seed'])
        random.seed(config['random_seed'])

        self.trainSize = len(dataModel.trainSet)
        self.testSize = len(dataModel.testSet)
        self.numUser = dataModel.numUser
        self.numItem = dataModel.numItem
        self.groundTruth = dataModel.groundTruth

        self.user_items_test = dataModel.user_items_test
        self.itemsInTestSet = dataModel.itemsInTestSet
        self.fileName = dataModel.fileName
        self.logger = dataModel.logger
        self.cost = tf.constant(0.0)
        self.min_loss = np.PINF
        self.loss_increas_count = 0
        self.save_path = config['save_path']
        self.load_model = config['load_model']
        self.saver = None
        self.best_Recall = 0
        self.best_NDCG = 0
        self.best_MRR=0

        self.best_Recall_BatchId = 0
        self.best_Recall_EpochId = 0

        self.optimizer = None
        self.sess = None


        self.fileName = config['fileName']
        self.learnRate = config['learnRate']
        self.maxIter = config['maxIter']
        self.trainBatchSize = config['trainBatchSize']
        self.testBatchSize = config['testBatchSize']
        self.topN = config['topN']
        self.early_stop = config['early_stop']
        self.eval_user_lists = self.generate_eval_user_lists()

        if self.trainSize % self.trainBatchSize == 0:
            self.trainBatchNum = int(self.trainSize // self.trainBatchSize)
        else:
            self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1

        if self.testSize % self.testBatchSize == 0:
            self.testBatchNum = int(self.testSize // self.testBatchSize)
        else:
            self.testBatchNum = int(self.testSize // self.testBatchSize) + 1

        "初始化的时候直接将test dataset中的正确label传进了eval类"
        self.evaluateResult = eval.Metrics.Metrics(groundTruthLists=self.groundTruth,
                                                topK=self.topN)
        # user/item embedding
        self.userEmbedding = None
        self.itemEmbedding = None

    def run(self):
        self.buildModel()
        self.trainModel()

    def buildModel(self):
        pass

    def trainModel(self):
        self.optimizer = tf.train.AdamOptimizer(self.learnRate, name='Adam_optimizer').minimize(self.cost)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if self.load_model:
            cur_save_path = './' + self.save_path + '/' + self.fileName
            ckpt = tf.train.get_checkpoint_state(cur_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print('model has been loaded\n')
            else:
                print('no model has been loaded\n')
                pass

        for epochId in range(self.maxIter):
            start = time.time()
            totalLoss = 0
            for batchId in range(self.trainBatchNum):
                loss = self.trainEachBatch(epochId, batchId)
                totalLoss += loss
            end = time.time()
            self.logger.info("time cost of an epoch:" + str(end - start) + ", totalLoss: " + str(totalLoss))

            if np.isnan(totalLoss):
                self.logger.info("the loss is nan, training stopped.")
                break
            if totalLoss < self.min_loss:
                self.min_loss = totalLoss

        tf.reset_default_graph()

    def trainEachBatch(self, epochId, batchId):
        pass

    def generate_eval_user_lists(self):
        eval_user_lists = []
        test_user_list = list(self.user_items_test.keys())
        idx_range = len(test_user_list)

        if idx_range % self.testBatchSize == 0:
            step_num = idx_range // self.testBatchSize
        else:
            step_num = idx_range // self.testBatchSize + 1

        for i in range(step_num):
            start = self.testBatchSize * i
            end = start + self.testBatchSize
            if end > idx_range:
                end = idx_range
                start = end - self.testBatchSize
            user_idices = test_user_list[start:end]
            "eval_user_lists中分testBatchSize存放了特定长度的userIdx子列表, 如果不够会从end往前数，造成有重复的"
            eval_user_lists.append(user_idices)
        return eval_user_lists

    def evaluate(self, epochId, batchId):

        userPredLists = {}

        packTime_total = 0
        runTime_total = 0
        sortTime_total = 0

        for user_list in self.eval_user_lists:
            "user_list中存放了userIdx子列表，shape 为 testBatchSize，user_pred_lists中存放了预测出的概率最高的10个itemIdx"
            user_pred_lists, packTime, runTime, sortTime = self.getPredList_ByUserIdxList(user_list)
            packTime_total += packTime
            runTime_total += runTime
            sortTime_total += sortTime
            "userPredLists[userIdx] 中存放了每个用户概率最高的十个itemIdx"
            "userPredList是个字典，重复的userIdx会对应于同一个key，消除了重复情况"
            for i in range(len(user_list)):
                userIdx = user_list[i]
                pred_list = user_pred_lists[i]
                userPredLists[userIdx] = pred_list

        "将预测出的TopK个item放入eval函数中，与在函数初始化时就已经存在的groundtruth比较计算指标" \
        "userPredLists是个字典"
        self.evaluateResult.setPredLists(userPredLists)
        newRecall, newMRR, newNDCG = self.evaluateResult.calMatrics()
        # save best result
        if newRecall > self.best_Recall:
            self.best_MRR = newMRR
            self.best_Recall = newRecall
            self.best_NDCG = newNDCG
            self.best_Recall_EpochId = epochId
            self.best_Recall_BatchId = batchId
        # show best result
        self.logger.info("\n----------------------------------------------------------------------\n")
        self.logger.info("best Recall result: %.4f, batchId: %d, epochId: %d" %
                         (self.best_Recall, self.best_Recall_BatchId, self.best_Recall_EpochId))
        self.logger.info("best MRR result: %.4f, batchId: %d, epochId: %d" %
                         (self.best_MRR, self.best_Recall_BatchId, self.best_Recall_EpochId))
        self.logger.info("best NDCG result: %.4f, batchId: %d, epochId: %d" %
                         (self.best_NDCG, self.best_Recall_BatchId, self.best_Recall_EpochId))
        self.logger.info("\n----------------------------------------------------------------------\n")


    def getPredList_ByUserIdxList(self, userIdx):
        pass




