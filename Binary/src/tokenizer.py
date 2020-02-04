from transformers import AutoTokenizer
from config import SetupParameters


class NERTokenizer():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(SetupParameters.TOKENIZER_ID, do_lower_case=True)

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, tok_ids):
        return self.tokenizer.decode(tok_ids)
