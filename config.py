import os

class SetupParameters():

    BERT_INPUT_LIMIT = 200#512

    # number of articles from the dataset (set to -1 to read the entire dataset)
    DATA_LIMIT = -1
    MODEL_ID = 'Musixmatch/umberto-commoncrawl-cased-v1'
    TOKENIZER_ID = 'Musixmatch/umberto-commoncrawl-cased-v1'
    SAVING_PATH = 'checkpoints/'
    LOAD_FILE = SAVING_PATH+'state_dict.pt'


class TrainingParameters():
    SEED = 24
    #OPTIMIZATION_STEPS = 128
    EPOCHS_NUM = 6
    BATCH_SIZE = 32
    BERT_LEARNING_RATE = 1e-6
    LINEAR_L_LEARNING_RATE = 1e-3
    # [training, validation, test]
    DATASET_SPLIT = [.7, .15, .15]
    SHUFFLE_DATASET = True
    WORKERS_NUM = 0
    # milestones at wich change the learning rate
    MILESTONE_STEPS = 2
    MILESTONES = list(range(0, EPOCHS_NUM, MILESTONE_STEPS))
    # factor to which multiply the learning rate at each milestone
    GAMMA = 0.5

    # labels frequencies:
    #   - O: 3031663
    #   - B-PER: 68199
    #   - I-PER: 45713
    #   - B-LOC: 128404
    #   - I-LOC: 72368
    #   - B-ORG: 22121
    #   - I-ORG: 12734
    #   - B-MISC: 37693
    #   - I-MISC: 42023
