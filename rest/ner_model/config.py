
class SetupParameters():

    BERT_INPUT_LIMIT = 300
    # TODO delete model id (from pretrained)
    # number of articles from the dataset (set to -1 to read the entire dataset)
    MODEL_SEED = 'Musixmatch/umberto-commoncrawl-cased-v1'
    TOKENIZER_ID = 'Musixmatch/umberto-commoncrawl-cased-v1'
    LOAD_PATH = 'pretrained_dict/state_dict.pt'