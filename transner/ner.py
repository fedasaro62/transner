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
from dateparser.search import search_dates
from simpletransformers.ner.ner_model import NERModel

from .utils import NERSeparatePunctuations

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
                    'I-MISC': 8,
                    'B-DOC': 9,
                    'I-DOC': 10,
                    'B-PROC': 11,
                    'I-PROC': 12
                    }
_LABEL_TO_TARGET = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'B-DOC', 'I-DOC', 'B-PROC', 'I-PROC']

_SHORT_TO_TYPE = {'PER': 'PERSON',
                'LOC': 'LOCATION',
                'ORG': 'ORGANIZATION',
                'MISC': 'MISCELLANEOUS',
                'DOC': 'DOCUMENT',
                'PROC': 'PROCEDURE'
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
                    'EU_PHONE_NUMBER': _CLEAN_START_REGEX + '([+]*[(]?[0-9]{1,4}[)]?){0,1}([\.0-9]+){10}' + _CLEAN_END_REGEX,
                    'EMAIL_ADDRESS': _CLEAN_START_REGEX + '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+' + _CLEAN_END_REGEX,
                    'IPV4_ADDRESS': _CLEAN_START_REGEX + '((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}' + _CLEAN_END_REGEX,
                    'URI': r'\[URL_[0-9]+\]'
                }
_RULE_BASED_SCORE = '0.9000' #keep it as a string to avoid problems



class Transner():

    def __init__(self, 
                pretrained_model, 
                use_cuda, 
                multi_gpu=False, 
                quantization=False, 
                cuda_device=-1, 
                language_detection=False, 
                threshold=0):
        """Transner object constructor

        Args:
            pretrained_model (str): Pretrained model name or path. If the model is not present locally it will be downloaded (if available)
            use_cuda (bool): flag to use gpu model
            quantization (bool, optional): Flag to use quantized model. Defaults to False.
            cuda_device (int, optional): Id of the gpu device to use. Defaults to -1.
            multi_gpu (bool, optional): Flag to use available all gpus (please set CUDA_DEVICE_ORDER and CUDA_VISIBLE_DEVICES env variable). Defaults to False.

            threshold (float, optional): threshold for filter the confidence
        """
        assert pretrained_model is not None, 'Pretrained model required'

        pretrained_path = self.get_model_path(pretrained_model)
        
        if language_detection:
            import fasttext
            self.get_model_detection_languages()
            self.language_detection_model = fasttext.load_model('lid.176.bin')
        args = {'no_cache': True, 
                'use_cached_eval_features': False,
                'max_seq_length': 512, #default=128
                'process_count': 1, 
                'silent': True, 
                'n_gpu': 1 if not multi_gpu else 2} #n_gpu > 1 means multi-gpu (the number of gpus depend on the environment)
        self.model = NERModel('bert', 
                            pretrained_path,
                            use_cuda=use_cuda,
                            args=args,
                            cuda_device=cuda_device)
        self.threshold = threshold
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


    def get_model_detection_languages(self):
        # https://fasttext.cc/docs/en/language-identification.html
        if not os.path.exists('lid.176.bin'):
            url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
            try:
                lang_model = wget.download(url)
            except:
                raise(Exception('Error while downloading the language detection model!'))


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
                            F.softmax(torch.tensor(logs).float(), dim=-1).max().item()
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
                                                'confidence': float(_RULE_BASED_SCORE),
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
                                            'confidence': float(_RULE_BASED_SCORE),
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
            sentence = re.sub(r'[^a-zA-Z0-9 ]', '', item['sentence'])
            try:
                langs_detected = self.language_detection_model.predict(sentence, k=1)
                self.language = re.sub('__label__', '', langs_detected[0][0])
                dates = search_dates(item['sentence'], languages=[self.language])
            except ValueError as e:
                continue
            if dates:
                starting_index = 0
                for date in dates:             
                    occurrence = re.search(re.escape(date[0]), item['sentence'][starting_index:])
                    time_type = self.check_opening_time(item['entities'])
                    try:
                        if not (item['sentence'][occurrence.start() - 1] == ' ' and item['sentence'][occurrence.end() + 1] == ' '):
                            if not self.find_overlap(item['entities'], occurrence.start(), occurrence.end()):
                                item['entities'].append(
                                {'type': time_type,
                                'value': date[0],
                                'confidence': _RULE_BASED_SCORE,
                                'offset': starting_index + occurrence.start()})
                            
                        starting_index = starting_index + occurrence.end()
                    except IndexError:
                        # the element is at the beginning or ending of the sentence
                        if occurrence.start() == 0 or occurrence.end() == len(item['sentence']):
                            if not self.find_overlap(item['entities'], occurrence.start(), occurrence.end()):
                                item['entities'].append(
                                {'type': time_type,
                                'value': date[0],
                                'confidence': float(_RULE_BASED_SCORE),
                                'offset': starting_index + occurrence.start()})
                            
                        starting_index = starting_index + occurrence.end()

        return ner_dict


    def find_overlap(self, entities, candidate_start, candidate_end):
        for entity in entities:
            entity_start, entity_end = entity['offset'], entity['offset'] + len(entity['value'])
            
            '''if candidate_start == entity_start or candidate_end == entity_end:
                return True

            if candidate_start >= entity_start and candidate_end <= entity_end:
                return True

            if candidate_start < entity_start and candidate_end < entity_end and candidate_end > entity_start:
                return True
            
            if candidate_start > entity_start and candidate_end > entity_end and candidate_start < entity_end:
                return True'''
       
            #1
            if candidate_start < entity_start and candidate_end < entity_end and candidate_end > entity_start:
                return True
            #2
            if candidate_start > entity_start and candidate_end < entity_end:
                return True
            #3
            if candidate_start > entity_start and candidate_end > entity_end and candidate_start < entity_end:
                return True
            #4
            if candidate_start == entity_start and candidate_end == entity_end:
                return True
            #5 - 5bis
            if candidate_start == entity_start or candidate_end == entity_start:
                return True
            #6 - 6bis
            if candidate_end == entity_end or candidate_start == entity_end:
                return True
            # 7
            if candidate_start < entity_start and candidate_end > entity_end:
                return True

        return False


    def check_opening_time(self, entities):
        loc_org_presence = False
        for entity in entities:
            if entity['type'] == 'ORGANIZATION' or entity['type'] == 'LOCATION':
                loc_org_presence = True
                break
                   
        if loc_org_presence:
            return 'T_OPENING'
        return 'TIME'

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
                                        'confidence': float(str(np.mean(active_e_scores))[:6]),
                                        'offset': beginning_offset}
                            # often the string "mario è" is tagged with a person. The following operation manually fixes this problem
                            if curr_entity['value'][-2:] == ' è':
                                curr_entity['value'] = curr_entity['value'][:-2]
                            if curr_entity['confidence'] >= self.threshold:
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
                                        'confidence': float(str(np.mean(active_e_scores))[:6]),
                                        'offset': beginning_offset}
                            if curr_entity['confidence'] >= self.threshold:
                                curr_res['entities'].append(curr_entity)
                            beginning_offset    = curr_offset
                            active_e_type       = e_type[2:]
                            active_e_value      = e_value + ' ' #here previous version was +=
                            active_e_scores     = [score]
                    elif e_type[0] == 'O' and active_e_type:
                        curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 
                                    'value': active_e_value[:-1], 
                                    'confidence': float(str(np.mean(active_e_scores))[:6]),#4 decimal digits
                                    'offset': beginning_offset}
                        # often the string "mario è" is tagged with a person. This operation clean this problem
                        if curr_entity['value'][-2:] == ' è':
                            curr_entity['value'] = curr_entity['value'][:-2] 
                        if curr_entity['confidence'] >= self.threshold:
                            curr_res['entities'].append(curr_entity)
                        beginning_offset    = None
                        active_e_type       = None
                        active_e_value      = ''
                        active_e_scores     = [score]

                    #if last prediction for that string, then save the active entities
                    if curr_offset >= len(s) and active_e_type:
                        curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 
                                    'value': active_e_value[:-1], 
                                    'confidence': float(str(np.mean(active_e_scores))[:6]),#4 decimal digits
                                    'offset': beginning_offset}
                        if curr_entity['confidence'] >= self.threshold:
                            curr_res['entities'].append(curr_entity)
                result_dict.append(curr_res)

            return result_dict
