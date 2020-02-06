from transformers import AutoTokenizer


class NERTokenizer():

    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, do_lower_case=True)

    def tokenize(self, text):
        #TODO add special tokens
        return self.tokenizer.encode(text)

    def detokenize(self, tok_ids):
        return self.tokenizer.decode(tok_ids)
