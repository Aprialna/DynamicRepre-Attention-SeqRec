
import numpy as np

class Metrics:

    def __init__(self, groundTruthLists, topK):

        '''
        :param groundTruthLists: a dict {userId:[next_item], ...}
        :param topK:    Integer
        '''
        self.groundTruthLists = groundTruthLists
        self.predLists = None
        self.topK = topK

    def setPredLists(self, predLists):
        self.predLists = predLists

    '''Matrics'''

    def calMatrics(self):
        recall = []
        mrr = []
        ndcg=[]
        for userIdx in self.groundTruthLists:
            m_recall, m_mrr, m_ndcg = self.calculate_a_matric(userIdx)
            recall.append(m_recall)
            mrr.append(m_mrr)
            ndcg.append(m_ndcg)
        RecallSum = 0
        MRRSum = 0
        NDCGSum = 0
        for result in recall:
            RecallSum += result
        for result in mrr:
            MRRSum += result
        for result in ndcg:
            NDCGSum += result

        return RecallSum/len(recall), MRRSum/len(mrr),NDCGSum/len(ndcg)

    def calculate_a_matric(self, userIdx):
        userTrueList = self.groundTruthLists[userIdx]
        userPredList = self.predLists[userIdx]
        hit = 0
        mrr = 0
        ndcg = 0
        if userTrueList[0] in userPredList:
            hit += 1
            rank = userPredList.index(userTrueList[0])
            if rank < self.topK:
                mrr += (1.0 / (rank + 1))
                ndcg += (1.0 / np.log2(rank + 2))
        else:
            mrr += 0
            ndcg += 0

        return hit,mrr,ndcg


