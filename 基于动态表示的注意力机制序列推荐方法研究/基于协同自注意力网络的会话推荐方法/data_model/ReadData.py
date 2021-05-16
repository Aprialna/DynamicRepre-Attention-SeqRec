import random
import logging
import os.path
import time

class ReadData:

    def __init__(self, config):
        self.fileName = config['fileName']
        self.inputPath = './dataset/processed_datasets/' + self.fileName
        self.outputPath = '../dataset/processed_datasets/' + self.fileName
        self.logger = self.initialize_logger('log')
        self.trainSet = []
        self.testSet = []
        self.trainSize = 0
        self.testSize = 0
        self.numUser = 0
        self.numItem = 0
        self.familiar_user_num = config['familiar_user_num']
        self.sample_size = config['sample_size']
        self.similarity = config['similarity']
        "都是字典，user_items_train, user_items_test"
        self.user_items_train = {}
        self.user_items_test = {}
        self.item_user_train = {}
        self.session_time = {}

        self.userIdToUserIdx = {}
        self.itemIdToItemIdx = {}
        self.userIdxToUserId = {}
        self.itemIdxToItemId = {}
        self.itemsInTestSet = set()
        self.itemIdxToPastUserIdx = {}
        self.itemIdxToPastUserTimePosition = {}

        random.seed(123)

    def buildModel(self):
        self.logger.info("\n###### information of DataModel ######\n")
        self.readData()


    def readData(self):
        trainPath = self.inputPath + '/train.txt'
        testPath = self.inputPath +  '/test.txt'
        fullPath = self.inputPath + '/full.txt'
        # read train data
        basicUserIdx = 1
        basicItemIdx = 0
        self.userIdToUserIdx[0] = 'pad'
        self.userIdxToUserId['pad'] = 0

        "添加pad item ， itemIdx为0"
        """
        self.itemIdxToItemId[0] = 'pad'
        self.itemIdToItemIdx['pad'] = 0
        self.itemIdxToItemInfor[0] = {'ItemFre': 0, 'path': [], 'code': [], 'len': 0}
        basicItemIdx += 1
        """
        user_in_testSet = set()
        record_list = []
        with open(fullPath) as fullFile:
            for line in fullFile:
                record = line.strip('\n').split(' ')
                record_list.append((record[0], record[1], float(record[2])))

            for records in record_list:
                userId, itemId, time = records[0], records[1], records[2]
                if userId not in self.userIdToUserIdx:
                    userIdx = basicUserIdx
                    self.userIdToUserIdx[userId] = userIdx
                    self.userIdxToUserId[userIdx] = userId
                    basicUserIdx += 1
                else:
                    userIdx = self.userIdToUserIdx[userId]

                if itemId not in self.itemIdToItemIdx:
                    itemIdx = basicItemIdx
                    afterPadItemIdx = itemIdx + 1
                    self.itemIdToItemIdx[itemId] = itemIdx
                    self.itemIdxToItemId[itemIdx] = itemId
                    self.itemIdxToPastUserIdx[afterPadItemIdx] = [userIdx]
                    self.itemIdxToPastUserTimePosition[afterPadItemIdx] = {userIdx: 0}
                    basicItemIdx += 1
                else:
                    itemIdx = self.itemIdToItemIdx[itemId]
                    afterPadItemIdx = itemIdx + 1
                    self.itemIdxToPastUserTimePosition[afterPadItemIdx][userIdx] = len(self.itemIdxToPastUserIdx[afterPadItemIdx])
                    self.itemIdxToPastUserIdx[afterPadItemIdx].append(userIdx)

        with open(trainPath) as trainFile:
            for line in trainFile:
                record = line.strip('\n').split(' ')
                userId, itemId, time = record[0], record[1], float(record[2])

                if userId not in self.userIdToUserIdx:
                    userIdx = basicUserIdx
                    self.userIdToUserIdx[userId] = userIdx
                    self.userIdxToUserId[userIdx] = userId
                    basicUserIdx += 1
                else:
                    userIdx = self.userIdToUserIdx[userId]

                if itemId not in self.itemIdToItemIdx:
                    itemIdx = basicItemIdx
                    self.itemIdToItemIdx[itemId] = itemIdx
                    self.itemIdxToItemId[itemIdx] = itemId
                    basicItemIdx += 1
                else:
                    itemIdx = self.itemIdToItemIdx[itemId]

                "实际计算时都是采用Idx计算的，而不是采用Id"
                self.user_items_train.setdefault(userIdx, [])
                self.item_user_train.setdefault(itemIdx + 1, [])
                self.user_items_train[userIdx].append(itemIdx)
                self.item_user_train[itemIdx + 1].append(userIdx)
                self.session_time.update({userIdx: time})
                self.trainSet.append([userIdx, itemIdx, time])

        # read test data
        with open(testPath) as testFile:
            for line in testFile:
                record = line.strip('\n').split(' ')
                userId, itemId, time = record[0], record[1], float(record[2])

                if userId not in self.userIdToUserIdx:
                    userIdx = basicUserIdx
                    self.userIdToUserIdx[userId] = userIdx
                    self.userIdxToUserId[userIdx] = userId
                    basicUserIdx += 1
                else:
                    userIdx = self.userIdToUserIdx[userId]

                if itemId not in self.itemIdToItemIdx:
                    itemIdx = basicItemIdx
                    self.itemIdToItemIdx[itemId] = itemIdx
                    self.itemIdxToItemId[itemIdx] = itemId
                else:
                    itemIdx = self.itemIdToItemIdx[itemId]

                user_in_testSet.add(userIdx)

                if userIdx not in self.user_items_test.keys():
                    self.user_items_test[userIdx] = []
                self.user_items_test[userIdx].append(itemIdx)
                self.testSet.append([int(userIdx), int(itemIdx), float(time)])
                self.itemsInTestSet.add(itemIdx)


        self.logger.info('Num user in testSet: ' + str(len(user_in_testSet)))

        self.trainSize = len(self.trainSet)
        self.testSize = len(self.testSet)
        self.numUser = len(self.userIdToUserIdx)
        self.numItem = len(self.itemIdToItemIdx)


        # print userIdx to userId
        userIdx_id_output_path = self.inputPath + '/user_idx_id.txt'
        userIdx_id_output_outputLines = []
        userIdx_id_output_outputLines.append("idx   id\n")
        for userId in self.userIdToUserIdx:
            userIdx = self.userIdToUserIdx[userId]
            userIdx_id_output_outputLines.append(str(userIdx) + "   " + str(userId) + "\n")
        with open(userIdx_id_output_path, 'w') as userIdx_id_output_file:
            userIdx_id_output_file.writelines(userIdx_id_output_outputLines)

        # print itemIdx to itemId
        itemIdx_id_output_path = self.inputPath + '/item_idx_id.txt'
        itemIdx_id_outputLines = []
        itemIdx_id_outputLines.append("idx  id\n")
        for itemId in self.itemIdToItemIdx:
            itemIdx = self.itemIdToItemIdx[itemId]
            itemIdx_id_outputLines.append(str(itemIdx) + "  " + str(itemId + "\n"))
        with open(itemIdx_id_output_path, 'w') as itemIdx_id_output_file:
            itemIdx_id_output_file.writelines(itemIdx_id_outputLines)


    def initialize_logger(self, output_dir):
        logger = logging.getLogger(name='hhh')

        logger.setLevel(logging.DEBUG)

        # create console handler and set level to info
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create error file handler and set level to error
        handler = logging.FileHandler(os.path.join(output_dir, "error.log"), "w", encoding=None, delay="true")
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create debug file handler and set level to debug
        handler = logging.FileHandler(os.path.join(output_dir, str(time.time()) + '.log'), "w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger










