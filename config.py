import os

class SetupParameters():

    BERT_INPUT_LIMIT = 300#512

    # number of articles from the dataset (set to -1 to read the entire dataset)
    DATA_LIMIT = -1
    MODEL_ID = 'Musixmatch/umberto-commoncrawl-cased-v1'
    TOKENIZER_ID = 'Musixmatch/umberto-commoncrawl-cased-v1'
    SAVING_PATH = 'checkpoints/'


class TrainingParameters():
    SEED = 24
    #OPTIMIZATION_STEPS = 128
    EPOCHS_NUM = 4
    BATCH_SIZE = 8
    BERT_LEARNING_RATE = 2e-5
    LINEAR_L_LEARNING_RATE = 4e-4
    # [training, validation, test]
    DATASET_SPLIT = [.7, .15, .15]
    SHUFFLE_DATASET = True
    WORKERS_NUM = 0
    # milestones at wich change the learning rate
    MILESTONES = list(range(0, EPOCHS_NUM, 5))
    # factor to which multiply the learning rate at each milestone
    GAMMA = 0.5
