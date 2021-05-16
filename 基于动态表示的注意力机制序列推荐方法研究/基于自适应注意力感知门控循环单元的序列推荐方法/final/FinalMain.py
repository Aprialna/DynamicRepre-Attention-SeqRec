# 取最后一个为Vj
import tensorflow as tf
from six.moves import cPickle
import os
import numpy as np
import random

from final.FinalEvaluate import Prediction
from final.FinalModel import Model
# from final.InputModel import Model
# from final.HiddenModel import Model
from utils import batch_sequences_lengths, zero_padding_three,zero_padding
import pandas as pd
from operator import itemgetter
batch_size = 10
lstm_size = 128
epochs = 3

#CA epoch=2
n_user = 2031
n_input = n_poi = 3112+1 # 补长度要用到最后一个ID，所以在总数上+1
#gowalla epoch=2
# n_user = 5073
# n_input = n_poi = 7021+1
#brightkite epoch =3
# n_user = 1850
# n_input = n_poi = 1672+1

save_dir = 'results'



if __name__=='__main__':
    train_root = '../data/CA/idxseq_train.txt'
    test_root = '../data/CA/idxseq_test.txt'
    popular_path = '../data/CA/popular.txt'
    top_k = [5, 10, 15, 20]

    # 获取用户对应的训练集和测试集，用户作为key，items作为value
    data_train = pd.read_csv(train_root, delimiter='\t', names=['user', 'items'])
    train_seq = dict(zip(data_train['user'], data_train['items'])) # train set中的user:[items]
    data_test = pd.read_csv(test_root, delimiter='\t', names=['user', 'items'])
    test_seq = dict(zip(data_test['user'], data_test['items']))# test set中的user:[items]
    n_batch = n_user // batch_size

    file = open(popular_path, 'r')
    popular_items = file.readline()
    popular_items = list(map(int, popular_items.strip('\t').split('\t')))
    file.close()

    train_user_pos_neg = []
    # [[user1,[histoty items],pos1,neg1],[user2,[hitory_items],pos2,neg2]……] train_user_pos_neg[0]=[user1,[histoty items],pos1,neg1],train_user_pos_neg[0][0] = user1,train_user_pos_neg[0][1]= [history items]
    for user in train_seq.keys():
        items_list = list(map(int,train_seq[user].strip(',').split(',')))
        pos_item = {}
        neg_item = {}
        input = {}
        for i in range(len(items_list)-1):
            input[i] = items_list[:i+1]
            pos_item[i] = items_list[i+1]
            neg = np.random.randint(n_input)
            while neg in items_list:
                neg = np.random.randint(n_input)
            neg_item[i] = neg
            train_user_pos_neg.append([user,input[i],pos_item[i],neg_item[i]])
    random.shuffle(train_user_pos_neg)
    with tf.Graph().as_default(), tf.Session() as sess:
        model = Model(batch_size, lstm_size, n_poi, n_user,is_training=True)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        print('Modeling Success!')
        for epoch in range(epochs):
            loss = []
            for b in range(n_batch):
                if ((b+1) * batch_size > len(train_user_pos_neg)):
                    high = len(train_user_pos_neg)
                else:
                    high = (b+1) * batch_size
                user_info = train_user_pos_neg[b * batch_size:high]

                u_pos = []
                u_neg=[]
                u_history=[]
                for k in range(len(user_info)): #这个userinfo里面是一个batch里面的所有history和pos和neg
                    u_history.append(user_info[k][1])
                    u_pos.append(user_info[k][2])
                    u_neg.append(user_info[k][3])
                max_length_history, hitory_lengths = batch_sequences_lengths(u_history) #padding有问题
                zero_padding(u_history, max_length_history, n_poi) #补得长度应该等于history的长度

                _, loss_sum, l2_loss = sess.run([model.train_op, model.cost, model.l2_loss],
                                                feed_dict={model.input_data: u_history, # 训练集的C
                                                           model.pos: u_pos,
                                                           model.neg: u_neg
                                                           })
                loss.append(loss_sum)
            checkpoint_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path)
            print('epoch:{}\tloss:{}'.format(epoch, np.mean(loss)))
    Prediction(save_dir,train_seq,test_seq,popular_items, n_poi, n_user,top_k)
