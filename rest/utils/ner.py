import pdb
import re

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
####################################
_CLEAN_START_REGEX = '(\s|^|[.,:])' 
_CLEAN_END_REGEX = '(\s|$|[.,])'
_REGEX_PATTERNS = {'IT_FISCAL_CODE': _CLEAN_START_REGEX + '[A-Z]{6}[0-9]{2}[A-E,H,L,M,P,R-T][0-9]{2}[A-Z0-9]{5}' + _CLEAN_END_REGEX,
                    'EU_IBAN': _CLEAN_START_REGEX + '[A-Z]{2}?[ ]?[0-9]{2}[]?[0-9]{4}[ ]?[0-9]{4}[ ]?[0-9]{4}[ ]?[0-9]{4}[ ]?[0-9]{4}' + _CLEAN_END_REGEX,
                    'NL_CITIZEN_SERVICE_NUMBER': _CLEAN_START_REGEX + '[0-9]{9}' + _CLEAN_END_REGEX ,
                    'UK_NATIONAL_ID_NUMBER': _CLEAN_START_REGEX + '[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]{1}[0-9]{6}[A-DFM]?' + _CLEAN_END_REGEX,
                    'EU_PHONE_NUMBER': _CLEAN_START_REGEX + '([+]*[(]?[0-9]{1,4}[)]?){0,1}([-\s\.0-9]+){10}' + _CLEAN_END_REGEX,
                    'EMAIL_ADDRESS': _CLEAN_START_REGEX + '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+' + _CLEAN_END_REGEX,
                    'IPV4_ADDRESS': _CLEAN_START_REGEX + '((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}' + _CLEAN_END_REGEX
                }


def make_ner_dict(input_strings, predictions):
    """This function produces a dictionary for the entities in predictions with the format {'sentence': ..., 'entities' :[{'type': , 'value': , 'offset':}]}
    
    Arguments:
        input_strings {list} -- list of the string from where the predictions come
        predictions {list} -- list of entities predicted for each input strings
    
    Returns:
        list -- list of dictionaries having the shape {'sentence': ..., 'entities' :[...]}
    """
    result_dict = []
    curr_res = {}
    for s, prediction in zip(input_strings, predictions):

        curr_res = dict()

        curr_res['sentence'] = s
        curr_res['entities'] = []
        curr_offset = 0
        #for multi-words entities
        beginning_offset = None
        active_e_type = None
        active_e_value = ''

        for e_pred in prediction:
            kv_pair = list(e_pred.items())
            assert len(kv_pair) == 1

            e_value, e_type = kv_pair[0]
            
            if e_type[0] == 'B':
                #if a entity is still active, close it
                if active_e_type:
                    curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 'value': active_e_value[:-1], 'offset': beginning_offset}
                    curr_res['entities'].append(curr_entity)
                    active_e_value = ''
                beginning_offset = curr_offset
                active_e_type= e_type[2:]
                active_e_value += e_value + ' ' #! active_e_value += re.sub(r'[^\w\s]', '', e_value)
            elif e_type[0] == 'I':
                #treat it as a beginner if the beginner is not present
                if not active_e_type:
                    beginning_offset = curr_offset
                    active_e_type = e_type[2:]
                    active_e_value += e_value + ' ' #! active_e_value += re.sub(r'[^\w\s]', '', e_value)
                elif e_type[2:] == active_e_type:
                    active_e_value += e_value + ' ' #! active_e_value += re.sub(r'[^\w\s]', '', e_value)
                else:
                    curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 'value': active_e_value[:-1], 'offset': beginning_offset}
                    curr_res['entities'].append(curr_entity)
                    beginning_offset = curr_offset
                    active_e_type = e_type[2:]
                    active_e_value += e_value + ' ' #! active_e_value += re.sub(r'[^\w\s]', '', e_value)
            elif e_type[0] == 'O' and active_e_type:
                curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 'value': active_e_value[:-1], 'offset': beginning_offset}
                curr_res['entities'].append(curr_entity)
                beginning_offset = None
                active_e_type = None
                active_e_value = ''

            #offset takes into account also the space
            curr_offset += len(e_value) + 1
            #pdb.set_trace()
            #if last prediction for that string, then save the active entities
            if curr_offset >= len(s) and active_e_type:
                curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 'value': active_e_value[:-1], 'offset': beginning_offset}
                curr_res['entities'].append(curr_entity)
        result_dict.append(curr_res)

    return result_dict





def find_from_regex(ner_dict):
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
                    
                    item['entities'].append({'type': field, 'value': matched_string, 'offset': offset})
    
    return ner_dict

