

class SetupParameters():

    BERT_INPUT_LIMIT = 321#512

    # CONTEXTUAL_SENTENCE_NUMBER is the number of sentence to take in consideration for the context
    # (e.g. set to 1 for single sentence context)
    CONTEXTUAL_SENTENCE_NUMBER = 1
    GILBERTO_MODEL = 'idb-ita/gilberto-uncased-from-camembert'
    GILBERTO_TOKENIZER = 'idb-ita/gilberto-uncased-from-camembert'


class TrainingParameters():
    SEED = 24
    EPOCHS_NUM = 4
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    # [training, validation, test]
    DATASET_SPLIT = [.7, .15, .15]
    SHUFFLE_DATASET = True
    RANDOM_SEED = 24
