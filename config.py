import os

class SetupParameters():

    BERT_INPUT_LIMIT = 300#512

    # CONTEXTUAL_SENTENCE_NUMBER is the number of sentence to take in consideration for the context
    # (e.g. set to 1 for single sentence context)
    CONTEXTUAL_SENTENCE_NUMBER = 1
    MODEL_ID = 'Musixmatch/umberto-commoncrawl-cased-v1'
    TOKENIZER_ID = 'Musixmatch/umberto-commoncrawl-cased-v1'
    SAVING_PATH = 'checkpoints/'


class TrainingParameters():
    SEED = 24
    #OPTIMIZATION_STEP = 128
    EPOCHS_NUM = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    # [training, validation, test]
    DATASET_SPLIT = [.7, .15, .15]
    SHUFFLE_DATASET = True
    WORKERS_NUM = 0
