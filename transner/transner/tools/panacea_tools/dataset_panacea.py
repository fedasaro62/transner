import os
import pdb
import re
import string

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Panacea(Dataset):
    """
    Class for the wiki NER dataset

    self.data = list of sentences.
    self.targets = list of NER labels per sentence.

    """

    def __init__(self, file_path):
        """
        Args:
            file_path: the path of the file for the wikiNER dataset
        """

        self.data = [] #list of sentences
        self.targets = [] #list of NER targets for each sentence

        raw_data , raw_targets = self.__read_data(file_path)
        self.__parse_sentences(raw_data, raw_targets)
        #self.__convert_to_BIO()



    def __read_data(self, file_path):

        data = []
        targets = []

        article_end = True
        just_started = True
        with open(file_path, 'r', encoding='UTF-8') as file:
            curr_words = []
            curr_labels = []

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
                
                if len(line.split()) != 2:
                    #pdb.set_trace()
                    #raise ValueError('[ERROR] Unknown file format')
                    continue

                curr_words.append(line.split()[0]) #word
                curr_labels.append(line.split()[1]) #NER label
            
        #add the last article
        if line != '\n':
            if len(curr_words) != len(curr_labels):
                raise ValueError('[ERROR] words-labels mismatch')
            data.append(curr_words)
            targets.append(curr_labels)

        return data, targets


    def __parse_sentences(self, source_data, source_targets):
        """
        Load the data from the source with the right format
        """
        curr_tags = []
        data = source_data
        targets = source_targets
        
        for count, (article, tags) in enumerate(zip(source_data, source_targets)):
            
            curr_tags = []
            curr_sentence = ''
            for word, tag in zip(article, tags):
                
                if word in ['.']:
                    self.data.append(curr_sentence + '.')
                    curr_tags.append(tag)
                    self.targets.append(curr_tags)
                    curr_sentence = ''
                    curr_tags = []
                    continue

                #if word in string.punctuation and word not in [',', '\'', '(', ')']:
                    #continue
                
                curr_sentence += word + ' '
                curr_tags.append(tag)
            if len(curr_sentence.split()) != len(curr_tags):
                raise ValueError("Sentence and target lengths do not match")
            #if SetupParameters.DATA_LIMIT != -1 and count >= SetupParameters.DATA_LIMIT:
                #break


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if index < 0:
            raise ValueError('[ERROR] fetching negative entry in the dataset')
        return self.data[index], self.targets[index]

    

    def get_max_sentence_len(self, data):
        curr_max = 0
        for item in data:
            item_tok_len = len(self.tokenizer.tokenize(item))
            if item_tok_len > curr_max:
                curr_max = item_tok_len

        return curr_max


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

