import os
import pdb
import re
import tarfile
import urllib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wget
from simpletransformers.ner.ner_model import NERModel

from .utils import NERSeparatePunctuations

from dateparser.search import search_dates

transner_folder     = '/'.join(__file__.split('/')[:-1])
default_models_path = os.path.join(transner_folder, 'models')
os.makedirs(default_models_path, exist_ok=True)
WORLD_CITIES_DB     = os.path.join(transner_folder, 'worldcities/worldcities.csv')
RELIGIONS_FILE      = os.path.join(transner_folder, 'religions.txt')


_TARGET_TO_LABEL = {'O': 0,
                    'B-PER': 1,
                    'I-PER': 2,
                    'B-LOC': 3,
                    'I-LOC': 4,
                    'B-ORG': 5,
                    'I-ORG': 6,
                    'B-MISC': 7,
                    'I-MISC': 8
                    }
_LABEL_TO_TARGET = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']

_SHORT_TO_TYPE = {'PER': 'PERSON',
                'LOC': 'LOCATION',
                'ORG': 'ORGANIZATION',
                'MISC': 'MISCELLANEOUS'
                }

########################################################################################################################################################
# List of regex to apply.
# The first two regex ensures that no substrings are matched, only strings starting with space, end of the sentence or final punctuations and  ending with the same.
# An example is a phone number like '1234567890', inside this also the NL_CITIZEN_SERVICE_NUMBER '123456789'could be matched otherwise.
########################################################################################################################################################
_CLEAN_START_REGEX = '(\s|^|[.,:])' 
_CLEAN_END_REGEX = '(\s|$|[.,])'
_REGEX_PATTERNS = {'IT_FISCAL_CODE': _CLEAN_START_REGEX + '[A-Z]{6}[0-9]{2}[A-E,H,L,M,P,R-T][0-9]{2}[A-Z0-9]{5}' + _CLEAN_END_REGEX,
                    'EU_IBAN': _CLEAN_START_REGEX + '[A-Z]{2}?[ ]?[0-9]{2}[]?[0-9]{4}[ ]?[0-9]{4}[ ]?[0-9]{4}[ ]?[0-9]{4}[ ]?[0-9]{4}' + _CLEAN_END_REGEX,
                    'NL_CITIZEN_SERVICE_NUMBER': _CLEAN_START_REGEX + '[0-9]{9}' + _CLEAN_END_REGEX ,
                    'UK_NATIONAL_ID_NUMBER': _CLEAN_START_REGEX + '[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]{1}[0-9]{6}[A-DFM]?' + _CLEAN_END_REGEX,
                    'EU_PHONE_NUMBER': _CLEAN_START_REGEX + '([+]*[(]?[0-9]{1,4}[)]?){0,1}([-\s\.0-9]+){10}' + _CLEAN_END_REGEX,
                    'EMAIL_ADDRESS': _CLEAN_START_REGEX + '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+' + _CLEAN_END_REGEX,
                    'IPV4_ADDRESS': _CLEAN_START_REGEX + '((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}' + _CLEAN_END_REGEX,
                    'URI': _CLEAN_START_REGEX + '\[URL_[0-9]+\]' + _CLEAN_END_REGEX
                }
_RULE_BASED_SCORE = 0.90



class Transner():

    def __init__(self, pretrained_model, use_cuda, quantization=False, cuda_device=-1):
        """Transner object constructor

        Args:
            pretrained_model (str): Pretrained model name or path. If the model is not present locally it will be downloaded (if available)
            use_cuda (bool): flag to use gpu model
            quantization (bool, optional): Flag to use quantized model. Defaults to False.
            cuda_device (int, optional): Id of the gpu device to use. Defaults to -1.
        """
        assert pretrained_model is not None, 'Pretrained model required'

        pretrained_path = self.get_model_path(pretrained_model)

        self.model = NERModel('bert', pretrained_path, use_cuda=use_cuda, args={'no_cache': True, 'use_cached_eval_features': False, 'process_count': 1, 'silent': True}, cuda_device=cuda_device)
        if cuda_device == -1 and quantization:
            #quantization currently available only for cpu
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
            supported_engines = torch.backends.quantized.supported_engines
            assert 'fbgemm' in supported_engines, 'FBGEMM is not a supported engine. Supported engines: {}'.format(supported_engines)
            torch.backends.quantized.engine='fbgemm'
            self.model.model = torch.quantization.quantize_dynamic(self.model.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.preprocesser = NERSeparatePunctuations()
        worlddb = pd.read_csv(WORLD_CITIES_DB)
        self.cities_set = set(worlddb['city'].str.lower())
        self.cities_set = self.cities_set.union(set(worlddb['city_ascii'].str.lower()))

        self.religions_set = set()
        file = open(RELIGIONS_FILE, 'r', encoding='UTF-8') 
        lines = file.readlines()
        for line in lines:
            if line.strip() != '':
                self.religions_set.add(line.strip().lower())


    def get_model_path(self, pretrained_model):
        """Get the path of the model: local path, from cache or search it in the cloud (if available)
        """
        #check if the model is a local path
        if os.path.exists(pretrained_model):
            return pretrained_model
        #otherwise check if model is present in the cache
        elif os.path.exists(os.path.join(default_models_path, pretrained_model)):
            return os.path.join(default_models_path, pretrained_model)
        #otherwise search it on the cloud
        else:
            url = 'http://venus.linksfoundation.com/transner_models/{}.tar.gz'.format(pretrained_model)
            try:
                pretrained_model = wget.download(url)
            except:
                raise(Exception('Model {} not found both locally and on the cloud'.format(pretrained_model)))
            #remove tar.gz at the end
            pretrained_model = '.'.join(pretrained_model.split('.')[:-2])
            tar = tarfile.open('{}.tar.gz'.format(pretrained_model))
            tar.extractall(path=default_models_path)
            tar.close()
            os.remove('{}.tar.gz'.format(pretrained_model))
            return os.path.join(default_models_path, pretrained_model)


    def reset_preprocesser(self):
        self.preprocesser.reset()


    def ner(self, input_strings, apply_regex=False, apply_gazetteers=False):
        """This function produces a dictionary for the recognized entities with the format:
            {'sentence': ..., 
            'entities' :[
                {'type': , 'value': , 'offset':},
                ...
                ]
            }
        
        Arguments:
            input_strings {list} -- list of the string from where the predictions come
            predictions {list} -- list of entities predicted for each input strings
        
        Returns:
            list -- list of dictionaries having the format {'sentence': ..., 'entities' :[...]}
        """

        processed_input = self.preprocesser.preprocess(input_strings, do_lower_case=True)
        # extract PER, LOC, ORG, MISC entity types
        (predictions, logits) = self.model.predict(processed_input)
        conf_scores = [
                        [
                            F.softmax(torch.tensor(logs), dim=-1).max().item()
                            for curr_item in curr_logits
                            for e_val, logs in curr_item.items()
                        ]
                        for curr_logits in logits
                    ]

        assert len(predictions) == len(input_strings), 'Batch sizes do not match'
        assert len(predictions) == len(conf_scores), 'Batch sizes do not match'

        ner_dict = self.make_ner_dict(processed_input, predictions, conf_scores)

        # post processing: get original strings (no preprocessed) and adjust the entities offset
        self.preprocesser.adjustEntitiesOffset([r['entities'] for r in ner_dict], adjust_case=True)
        for r, original in zip(ner_dict, input_strings):
            r['sentence'] = original
        if apply_regex:
            ner_dict = self.find_from_regex(ner_dict)
        if apply_gazetteers:
            ner_dict = self.find_from_gazetteers(ner_dict)

        return ner_dict


    def find_from_regex(self, ner_dict):
        """Find matches of regex patterns from ner_dict and insert the new entities found by means of the regex.
        
        Arguments:
            ner_dict {list} -- list of ner dictionaries where to insert new regex-found entities
        """

        for item in ner_dict:
            for field, regex in _REGEX_PATTERNS.items():
                for match in re.finditer(regex, item['sentence']):
                #   match = re.search(regex, item['sentence'])
                    if match:
                        matched_string = match.group(0)
                        offset = match.span(0)[0]
                        # remove initial or final space/punctuation if it was catched by regex
                        if matched_string[0] in '.,: ':
                            matched_string = matched_string[1:]
                            offset += 1
                        if matched_string[-1] in '., ':
                            matched_string = matched_string[:-1]
                        
                        item['entities'].append({'type': field,
                                                'confidence': round(_RULE_BASED_SCORE, 2),
                                                'value': matched_string, 
                                                'offset': offset})
        
        return ner_dict


    def find_from_gazetteers(self, ner_dict):
    
        # check religions
        for item in ner_dict:
            words_list = item['sentence'].lower().split()
            for word in words_list:
                if word in self.religions_set:
                    offset = item['sentence'].lower().index(word)
                    item['entities'].append({'type': 'RELIGION', 
                                            'value': item['sentence'][offset:offset+len(word)], 
                                            'confidence': round(_RULE_BASED_SCORE, 2),
                                            'offset': offset})

        # check nested LOC in MISCELLANEOUS 
        for item in ner_dict:
            for entity in item['entities']:
                if entity['type'] == 'MISCELLANEOUS':
                    #generate substrings and check if there is at least one contained in gazeetters
                    words_list = entity['value'].lower().split()
                    substrings = [words_list[i: j] for i in range(len(words_list)) for j in range(i + 1, len(words_list) + 1)]
                    for substring in substrings:
                        curr_str = ' '.join(substring)
                        if curr_str in self.cities_set:
                            offset = entity['value'].lower().index(curr_str)
                            item['entities'].append({'type': 'LOC', 
                                                    'value': entity['value'][offset:offset+len(curr_str)], 
                                                    'confidence': round(_RULE_BASED_SCORE, 2),
                                                    'offset': offset+entity['offset']})

        return ner_dict

    def find_dates(self, ner_dict):

        for item in ner_dict:
            dates = search_dates(item['sentence'])

            if dates:
                for date in dates:
                    for occurrence in re.finditer(date[0], item['sentence']):
                        item['entities'].append(
                            {'type': 'TIME',
                            'value': date[0],
                            'confidence': _RULE_BASED_SCORE,
                            'offset': occurrence.start()})

        return ner_dict

    def make_ner_dict(self, strings, predictions, conf_scores):
            """[summary]

            Arguments:
                strings {string} -- strings
                predictions {dict} -- dictionary of predictions

            Returns:
                dict -- format: 
                {'sentence': ..., 
                'entities' :[
                    {'type': , 'value': , 'offset':},
                    ...
                    ]
                }
            """
            result_dict = []
            curr_res    = {}
            for s, prediction, scores in zip(strings, predictions, conf_scores):
                assert len(prediction) == len(scores), 'Prediction and scores size mismatch'
                curr_res                = dict()
                curr_res['sentence']    = s
                curr_res['entities']    = []
                curr_offset             = 0
                #for multi-words entities
                beginning_offset        = None
                active_e_type           = None
                active_e_value          = ''
                active_e_scores         = []

                for e_pred, score in zip(prediction, scores):
                    kv_pair         = list(e_pred.items())
                    e_value, e_type = kv_pair[0]
                    curr_offset = curr_offset + s[curr_offset:].find(e_value) #flexible with more than one leading white spaces
                    if e_type[0] == 'B':
                        #if a entity is still active, close it
                        if active_e_type:
                            curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 
                                        'value': active_e_value[:-1],
                                        'confidence': round(np.mean(active_e_scores), 2),
                                        'offset': beginning_offset}
                            # often the string "mario è" is tagged with a person. The following operation manually fixes this problem
                            if curr_entity['value'][-2:] == ' è':
                                curr_entity['value'] = curr_entity['value'][:-2] 
                            curr_res['entities'].append(curr_entity)
                            active_e_value  = ''
                            active_e_scores = []
                        beginning_offset = curr_offset
                        active_e_type    = e_type[2:]
                        active_e_value   += e_value + ' ' #! assumption: always only one whitespace inside multi-word entities
                        active_e_scores.append(score)
                    elif e_type[0] == 'I':
                        #treat it as a beginner if the beginner is not present
                        if not active_e_type:
                            beginning_offset    = curr_offset
                            active_e_type       = e_type[2:]
                            active_e_value      += e_value + ' '
                            active_e_scores.append(score)
                        elif e_type[2:] == active_e_type:
                            active_e_value += e_value + ' '
                            active_e_scores.append(score)
                        else:
                            curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 
                                        'value': active_e_value[:-1],
                                        'confidence': round(np.mean(active_e_scores), 2),
                                        'offset': beginning_offset}
                            curr_res['entities'].append(curr_entity)
                            beginning_offset    = curr_offset
                            active_e_type       = e_type[2:]
                            active_e_value      = e_value + ' ' #here previous version was +=
                            active_e_scores     = [score]
                    elif e_type[0] == 'O' and active_e_type:
                        curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 
                                    'value': active_e_value[:-1], 
                                    'confidence': round(np.mean(active_e_scores), 2),
                                    'offset': beginning_offset}
                        # often the string "mario è" is tagged with a person. This operation clean this problem
                        if curr_entity['value'][-2:] == ' è':
                            curr_entity['value'] = curr_entity['value'][:-2] 
                        curr_res['entities'].append(curr_entity)
                        beginning_offset    = None
                        active_e_type       = None
                        active_e_value      = ''
                        active_e_scores     = [score]

                    #if last prediction for that string, then save the active entities
                    if curr_offset >= len(s) and active_e_type:
                        curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 
                                    'value': active_e_value[:-1], 
                                    'confidence': round(np.mean(active_e_scores), 2),
                                    'offset': beginning_offset}
                        curr_res['entities'].append(curr_entity)
                result_dict.append(curr_res)

            return result_dict
