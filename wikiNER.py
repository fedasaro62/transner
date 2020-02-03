import os
import pdb
import re
import string

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

from config import SetupParameters


class WikiNER(Dataset):
    """
    Class for the wiki NER dataset

    self.data = list of sentences.
    self.targets = list of NER labels per sentence.

    """

    _TARGET_TO_LABEL = {'O': 0,
                        'B-PER': 1,
                        'I-PER': 2,
                        'B-LOC': 3,
                        'I-LOC': 4,
                        'B-ORG': 5,
                        'I-ORG': 6,
                        'B-MISC': 7,
                        'I-MISC': 8}

    def __init__(self, file_path):
        """
        Args:
            file_path: the path of the file for the wikiNER dataset
        """

        self.data = []
        self.targets = []
        self.labels = []

        raw_data , raw_targets = self.__read_data(file_path)
        self.__parse_sentences(raw_data, raw_targets)
        self.__convert_to_BIO()
        self.__create_labels()

        self._item_len_limit = SetupParameters.BERT_INPUT_LIMIT
        #self.tokenizer = AutoTokenizer.from_pretrained(SetupParameters.TOKENIZER_ID, do_lower_case=True)
        



    def __read_data(self, file_path):

        data = []
        targets = []


        article_end = True
        just_started = True
        with open(file_path, 'r') as file:

            for line in file:
                if line == '\n':
                    article_end = True
                    continue

                if article_end:
                    # when the scan is just started there are no values inside curr_words and curr_labels
                    if not just_started:
                        if len(curr_words) != len(curr_labels):
                            raise ValueError('[ERROR] words-labels mismatch')
                        data.append(curr_words)
                        targets.append(curr_labels)
                    just_started = False
                    curr_words = []
                    curr_labels = []
                article_end = False
                for token in line.split():
                    triplet = token.split(r'|')
                    if len(triplet) != 3:
                        raise ValueError('[ERROR] Unknown file format')

                    curr_words.append(triplet[0]) #word
                    curr_labels.append(triplet[2]) #NER label

        return data, targets


    def __parse_sentences(self, source_data, source_targets):
        """
        Load the data from the source with the right format
        """
        curr_tags = []
        data = source_data
        targets = source_targets
        
        for count, (article, tags) in enumerate(zip(source_data, source_targets)):
            
            curr_sentence = ''
            for word, tag in zip(article, tags):
                
                if word in ['.']:
                    self.data.append(curr_sentence + '.')
                    curr_tags.append(tag)
                    self.targets.append(curr_tags)
                    curr_sentence = ''
                    curr_tags = []
                    continue

                if word in string.punctuation and word not in [',', '\'', '(', ')']:
                    continue
                
                curr_sentence += word + ' '
                curr_tags.append(tag)

            if SetupParameters.DATA_LIMIT != -1 and count >= SetupParameters.DATA_LIMIT:
                break

                    

    def __create_labels(self):
        for tags in self.targets:
            curr_tags = []
            for tag in tags:
                curr_tags.append(self._TARGET_TO_LABEL[tag])
            self.labels.append(curr_tags)       



    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if index < 0:
            raise ValueError('[ERROR] fetching negative entry in the dataset')
        

        item, labels = (self.data[index], self.labels[index])
        tok_item, tok_labels = self.tokenize_and_preserve_labels(item, labels, self.tokenizer)

        item_ids = self.tokenizer.convert_tokens_to_ids(tok_item)

        #here truncate if len+2 > 512. (+2 because of <s> and </s>)
        if len(item_ids)+2 > self._item_len_limit:
            item_ids = item_ids[:self._item_len_limit-2]
            tok_labels = tok_labels[:self._item_len_limit-2]
            #pdb.set_trace()

        # append <s> and </s> (5 and 6) at the beginning and end of the tensor. Then adjust the labels with the O         item_ids = self.tokenizer.build_inputs_with_special_tokens(item_ids)
        item_ids = self.tokenizer.build_inputs_with_special_tokens(item_ids)
        tok_labels.insert(0, 0)
        tok_labels.append(0)
        if len(item_ids) != len(tok_labels):
            #pdb.set_trace()
            raise ValueError('[ERROR] sentence and labels lengths do not match')
        
        item_tensor = torch.tensor(item_ids)
        labels_tensor = torch.tensor(tok_labels)
        attention_mask = torch.ones(len(item_tensor))

        to_pad = self._item_len_limit - len(item_ids)

        item_tensor = F.pad(item_tensor, pad=(0, to_pad), mode='constant', value=0)
        labels_tensor = F.pad(labels_tensor, pad=(0, to_pad), mode='constant', value=0)
        attention_mask = F.pad(attention_mask, pad=(0, to_pad), mode='constant', value=0)

        if len(item_tensor) != len(labels_tensor) or len(item_tensor) != len(attention_mask):
            raise ValueError('[ERROR] Tensors sizes do not match')
        #pdb.set_trace()
        return item, item_tensor, labels_tensor, attention_mask

    
    def tokenize_and_preserve_labels(self, sentence, text_labels, tokenizer):

        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword. It is, of course, a bit slower in processing
        time, but it will help our model achieve higher accuracy.
        """

        tokenized_sentence = []
        labels = []
        labels_of_beginners = [self._TARGET_TO_LABEL['B-PER'],
                            self._TARGET_TO_LABEL['B-LOC'],
                            self._TARGET_TO_LABEL['B-ORG'],
                            self._TARGET_TO_LABEL['B-MISC']]
        #punct_regex = r"[!\"#$%&\'\(\)\*\+,-./:;<=>\?@[\\]^_`{\|}~]"
        #punct_regex = "[.,\/#!$%\^&\*;:{}=\-_`~()]"

        # split by white space and punctuations (e.g. "Io parlo cinese, inglese e francese." = ['Io', 'parlo', 'cinese', ',', 'inglese', 'e', 'francese', '.'])
        for index, word in enumerate(sentence.split()):
            
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
            # Add the same label to the new list of labels `n_subwords` times
            if n_subwords > 1 and text_labels[index] in labels_of_beginners:
                # avoid to have multiple beginning [B-LOC, B-LOC, I-LOC, O, O] when BERT splits the beginning word
                labels.extend([text_labels[index]])
                labels.extend([text_labels[index]+1] * (n_subwords-1))
            else:
                labels.extend([text_labels[index]] * n_subwords)


        if len(tokenized_sentence) != len(labels):
            #pdb.set_trace()
            raise ValueError('[ERROR] sentence and labels lenghts do not match')
        #pdb.set_trace()
        return tokenized_sentence, labels


    def get_labels(self):
        """
        Returns the possible types for the entities
        """

        labels = set()
        for l in self.targets:
            for w in l:
                labels.add(w)

        return labels

    
    def get_max_sentence_len(self, data):
        curr_max = 0
        for item in data:
            item_tok_len = len(self.tokenizer.tokenize(item))
            if item_tok_len > curr_max:
                curr_max = item_tok_len

        return curr_max


    def get_over_limit(self, data, limit):
        """
        Returns the sentences exceeding the BERT input limit
        """
        s_list = []
        for item in data:
            if len(self.tokenizer.tokenize(item)) > limit:
                s_list.append(item)
        return s_list


    def __convert_to_BIO(self):
        """
        This method converts the wikiNER dataset to BIO notation
        """
 
        for article_num, tags in enumerate(self.targets):
            prev_tag = 'O'
            for tag_num, curr_tag in enumerate(tags):

                if curr_tag != 'O':
                    
                    if prev_tag == 'O' or prev_tag[1:] != curr_tag[1:]:
                        #here put B
                        self.targets[article_num][tag_num] = 'B' + curr_tag[1:]
                    
                prev_tag = curr_tag
