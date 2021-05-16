from data_model.GenerateInput import GenerateInput
from recommender.ACoSAN_model import ACoSAN_model
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':

    config = {
        'generate_seq': True,
        'fileName': 'yoo64',
        'dropout_rate': 0.8,#
        'early_stop': True,
        'random_seed': 123,
        'input_length': 6,#
        'learnRate': 0.001,
        'maxIter': 20,
        'trainBatchSize': 512,
        'testBatchSize': 512,
        'numFactor': 512,#
        'cell_numbers': 512,#
        'topN': 5,
        'keep_prob_user': 0.8,
        'familiar_user_num': 10,
        'sample_size':500,
        'load_model': False,
        'similarity': 'cosine',
        'save_path': 'saved_model'
    }


    dataModel = GenerateInput(config)
    dataModel.buildModel()
    input_len = config['input_length']
    dataModel.generate_sequences_hor(input_len)
    recommender = ACoSAN_model(dataModel, config)
    recommender.run()



























