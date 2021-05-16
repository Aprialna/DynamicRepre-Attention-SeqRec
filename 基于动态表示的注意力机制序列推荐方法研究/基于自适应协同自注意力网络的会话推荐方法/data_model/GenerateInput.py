from math import sqrt

from data_model.ReadData import ReadData
import random
from operator import itemgetter

class GenerateInput(ReadData):

    def __init__(self, config):
        super(GenerateInput, self).__init__(config)

        """
            self.train_users：一个batch中所有的user id。不妨假设一个batch只有一个用户
            self.train_sequences_input：表示用户的交互item的前n-1项
            self.train_sequences_user_input：表示根据每个item找到的user（邻居）
            self.train_sequences_user_weight：邻居的权重
            self.train_sequences_target：表示用户交互item的后n-1项
            self.train_sequences_negative：表示生成的负样本
            ### 训练时想用到目标信息，因此将用户交互item的最后一项作为计算adaptive attention的target
            ### self.train_last_item = [] # 每个batch有一个这样的item（每个用户交互序列的最后一个）
            ### self.train_last_nei = [] # 最后一个item也有很多的neighbor和weight
            self.user_pred_sequences：预测阶段的输入
            self.user_pred_user_sequences：根据测试输入得到的每个item对应的邻居
            self.user_pred_user_sequences_weight：测试邻居权重
            ### 测试时若将所有item作为target然后取平均或最大值，结果比不用目标信息差

        """
        self.train_users = []
        self.train_sequences_input = []
        self.train_sequences_user_input = []
        self.train_sequences_user_weight = []
        self.train_sequences_target = []
        self.train_sequences_negative = []
        # 新增
        self.train_last_item = []
        self.train_last_nei = []
        self.train_last_weight = []

        self.user_pred_sequences = {}
        self.user_pred_user_sequences = {}
        self.user_pred_user_sequences_weight = {}
        # 新增
        self.test_target_item = {}
        self.test_target_nei = {}
        self.test_target_weight = {}

    def buildModel(self):
        self.readData()
        self.append_pad_item()

    def append_pad_item(self):
        # add a padding item 0,所有的itemIdx+1，为了获得neighbor
        self.numItem += 1
        for userIdx, items in self.user_items_train.items():
            new_items = []
            for item in items:
                new_items.append(item+1)
            self.user_items_train[userIdx] = new_items

        for userIdx, items in self.user_items_test.items():
            new_items = []
            for item in items:
                new_items.append(item+1)
            self.user_items_test[userIdx] = new_items

    def generate_sequences_hor(self, input_length):
        self.groundTruth = {}
        self.logger.info("input length: %d" %input_length)

        "用一个slide window来在序列上滑行，slide window的size包括用来预测的input_length,和label的size，为target_size"
        "将原本用于序列的方法改成seq2seq，看看结果是怎么样的"
        seq_length = input_length
        # build train seqences
        for userIdx, items in self.user_items_train.items():
            for seq in self.slide_window(items, seq_length):
                item_past_userlist = []
                item_past_userweight =[]
                last_userlsit = []
                last_userweight = []

                # input_seq = seq[0:input_length-1] ## 现在最后一个也要计算neighbor，来构建adaptive target的表示
                target_seq = seq[1:input_length]
                history_before_t = []
                for i, train_item in enumerate(seq):
                    if train_item == 0:
                        item_past_user = [0] * self.familiar_user_num
                        item_past_weight = [0] * self.familiar_user_num
                    else:
                        history_before_t.append(train_item)
                        idx = 0
                        item_past_user = [0] * self.familiar_user_num
                        item_past_weight = [0] * self.familiar_user_num
                        neighbor_similarity = self.find_neighbors(train_item, self.item_user_train, self.user_items_train,
                                                             self.session_time,history_before_t,"recent", self.sample_size, self.familiar_user_num)
                        for temp_tuple in neighbor_similarity:
                            item_past_user[idx] = temp_tuple[0]
                            item_past_weight[idx]=temp_tuple[1]
                            idx+=1
                    if i == len(seq) - 1:  # 最后一个item
                        last_userlsit.append(item_past_user)
                        last_userweight.append(item_past_weight)
                    else:
                        item_past_userlist.append(item_past_user)
                        item_past_userweight.append(item_past_weight)
                        "shape: [input_size, familiar_user_num]"

                self.train_users.append(userIdx)
                "train_users 和 numUser不一样， numUser和testSize比较相近"
                self.train_sequences_input.append(seq[:seq_length-1])
                self.train_sequences_user_input.append(item_past_userlist)
                self.train_sequences_user_weight.append(item_past_userweight)
                self.train_sequences_target.append(target_seq)
                self.train_last_item.append([seq[-1]]) # 每个序列的最后一个item
                self.train_last_nei.append(last_userlsit) # [batch*1] 实际上一个维度上只有一个item
                self.train_last_weight.append(last_userweight)


        # build pred sequences
        for userIdx in self.user_items_test.keys():
            items = self.user_items_test[userIdx][:-1]
            groundTruth = self.user_items_test[userIdx][-1]
            "当items的数量不够时，使用item 0来进行填充"
            if len(items) < input_length-1:
                pred_seq = [0] * (input_length-1)
                pred_seq[-len(items):] = items
            else:
                pred_seq = items[-(input_length-1):]

            item_past_userlist = []
            item_past_userweight = []
            test_history_before_t = []
            for test_item in pred_seq:
                if test_item == 0:
                    item_past_user = [0] * self.familiar_user_num
                    item_past_weight = [0]*self.familiar_user_num
                else:
                    idx = 0
                    item_past_user = [0] * self.familiar_user_num
                    item_past_weight = [0] * self.familiar_user_num
                    neighbor_similarity = self.find_neighbors(test_item, self.item_user_train, self.user_items_train,
                                                            self.session_time, test_history_before_t,"recent", self.sample_size,
                                                            self.familiar_user_num)
                    for temp_tuple in neighbor_similarity:
                        item_past_user[idx]=temp_tuple[0]
                        item_past_weight[idx] = temp_tuple[1]
                        idx+=1
                item_past_userlist.append(item_past_user)
                item_past_userweight.append(item_past_weight)
                "shape: [input_size, familiar_user_num]"
            # target item
            target_userlist = []
            target_userweight = []
            if groundTruth == 0:
                target_nei = [0] * self.familiar_user_num
                target_weight = [0] * self.familiar_user_num
            else:
                idx = 0
                target_nei = [0] * self.familiar_user_num
                target_weight = [0] * self.familiar_user_num
                neighbor_similarity = self.find_neighbors(groundTruth, self.item_user_train, self.user_items_train,
                                                          self.session_time, test_history_before_t,"recent", self.sample_size,
                                                          self.familiar_user_num)
                for temp_tuple in neighbor_similarity:
                    target_nei[idx] = temp_tuple[0]
                    target_weight[idx] = temp_tuple[1]
                    idx += 1
            target_userlist.append(target_nei)
            target_userweight.append(target_weight)

            self.user_pred_sequences[userIdx] = pred_seq
            self.user_pred_user_sequences[userIdx] = item_past_userlist
            self.user_pred_user_sequences_weight[userIdx] = item_past_userweight
            self.groundTruth[userIdx] = [groundTruth]
            self.test_target_item[userIdx] = [groundTruth]
            self.test_target_nei[userIdx] = target_nei
            self.test_target_weight[userIdx] = target_weight

    def slide_window(self, itemList, window_size):

        if len(itemList) < window_size:
            seq = [0] * window_size
            seq[-len(itemList):] = itemList
            yield seq
        else:
            num_seq = len(itemList) - window_size + 1
            for startIdx in range(num_seq):
                endIdx = startIdx + window_size
                seq = itemList[startIdx:endIdx]
                yield seq

    def find_neighbors(self,input_item_id, item_session_map, session_item_map, session_time,history_before_t, sampling, sample_size, k):

        possible_neighbors = self.possible_neighbor_sessions(input_item_id, item_session_map, session_time, sample_size,
                                                        sampling)  # 根据item得到包含该item的session
        possible_neighbors = self.calc_similarity(history_before_t,possible_neighbors,
                                             session_item_map)  # 根据item得到其neighbor session和相似性

        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])  # 按相似性由大到小排序
        possible_neighbors = possible_neighbors[:k]  # neighbor session中取出最相似的session

        return possible_neighbors

    def possible_neighbor_sessions(self,input_item_id, item_session_map, session_time, sample_size, sampling):

        relevant_sessions = list(set(item_session_map[input_item_id]))
        if len(relevant_sessions) > sample_size:
            if sampling == 'recent':
                sample = self.most_recent_sessions(relevant_sessions, sample_size, session_time)
            elif sampling == 'random':
                sample = random.sample(relevant_sessions, sample_size)
            else:
                sample = relevant_sessions[:sample_size]

            return sample
        else:
            return relevant_sessions

    def calc_similarity(self,history_before_t,sessions, session_item_map):
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            session_items_test = session_item_map[session]  # 根据neighbor的session找到相关的item

            similarity = getattr(self , self.similarity)(set(session_items_test),set(history_before_t))
            # 通过neighbor session与current session的items求解两个session之间的相似性
            if similarity > 0:
                neighbors.append([session, similarity])

        return neighbors  # 返回neighbor session和其余当前session之间的相似性

    def most_recent_sessions(self,sessions, number, session_time):
        sample = set()

        tuples = list()
        for session in sessions:
            time = session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    # 用于计算item之间相似度的方法
    def jaccard(self,first, second):
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union
        return res


    def cosine(self, first, second):
        li = len(first & second)  # 交集
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def tanimoto(self, first, second):
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / (la + lb - li)

        return result

    def binary(self, first, second):
        a = len(first & second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result





