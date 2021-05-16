import numpy as np

def batch_sequences_lengths(batch_sequences):
    lengths = []
    max_length = 0
    for seq in batch_sequences:
        lengths.append(len(seq))
        if len(seq) > max_length:
            max_length = len(seq)
    return max_length, lengths

# three dimension padding and each dimension need to be padded
def zero_padding_three(sequences, target_seq_length,n_poi):
    max = 0
    for index, sub_seq in enumerate(sequences):
        sub_seq_length = len(sub_seq)
        sub_target_length = len(sub_seq[-1])
        if sub_target_length>max:
            max = sub_target_length
        if sub_seq_length < target_seq_length:
            minus_length = target_seq_length - sub_seq_length
            for i in range(minus_length):
                sequences[index].append([n_poi-1]) # 要补的元素，[n_poi-1]补得是最后一个item
    for index, sub_seq in enumerate(sequences):
        for subindex, subsub_seq in enumerate(sub_seq):
            subsub_seq_length = len(subsub_seq)
            if subsub_seq_length < max:
                minus_length2 = max - subsub_seq_length
                plus_sequence = [n_poi - 1] * minus_length2  # 要补的元素，[n_poi-1]补得是最后一个item
                sub_seq[subindex] += plus_sequence

# two dimension padding and each dimension meed to be padded
def zero_padding(sequences, target_seq_length,n_poi):
    for index, sub_seq in enumerate(sequences):
        sub_seq_length = len(sub_seq)
        if sub_seq_length < target_seq_length:
            minus_length = target_seq_length - sub_seq_length
            plus_sequence = [n_poi-1]*minus_length # 要补的元素，[n_poi-1]补得是最后一个item
            sequences[index] += plus_sequence

def user_padding(users, max_length):
    users_seq = []
    for index in range(len(users)):
        user = users[index]
        user_seq = np.full(max_length, user, dtype='int32')
        users_seq.append(user_seq)
    return users_seq



