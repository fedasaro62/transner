import getopt
import json
import pdb
import sys
import time

import torch

from model import BertNER
from tokenizer import NERTokenizer

_TARGET_TO_LABEL = {'O': 0,
                    'B-PER': 1,
                    'I-PER': 2,
                    'B-LOC': 3,
                    'I-LOC': 4,
                    'B-ORG': 5,
                    'I-ORG': 6,
                    'B-MISC': 7,
                    'I-MISC': 8}
_LABEL_TO_TARGET = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']

_SHORT_TO_TYPE = {'PER': 'PERSON',
                'LOC': 'LOCATION',
                'ORG': 'ORGANIZATION',
                'MISC': 'MISCELLANEOUS' 
                }


"""
main.py -f <state_dict_file> -s <string>
"""

def main(argv):
 
    dictfile = ''
    input_strings = []
    try:
        opts, args = getopt.getopt(argv,"hf:",["help=", "dictfile="])
    except getopt.GetoptError:
        print ('main.py -f <state_dict_file> <string1> ... <stringN>')
        sys.exit(2)
    for count, (opt, arg) in enumerate(opts):
        if opt in ('-h', '--help'):
            print ('Correct format: main.py -f <state_dict_file> <string1> ... <stringN>')
            sys.exit()
        elif opt in ("-f", "--dictfile"):
            dictfile = arg
        else:
            print('-h for help')
            sys.exit(2)

    for arg in args[count:]:
        input_strings.append(arg)
    
    if dictfile == '' or len(input_strings) == 0:
        print('Missing argument (-h for help)')
        sys.exit(2)
    

    # load the model and the tokenizer
    model = BertNER()
    state_dict = torch.load(dictfile)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = NERTokenizer()
    
    results_l = []
    for s in input_strings:
        entities_list = ner(model, tokenizer, text=s)
        results_l.append({'sentence': s, 'entities': entities_list})
    
    output = {'timestamp': time.time(), 'results': results_l}
    json_output = json.dumps(output, ensure_ascii=False)
    return json_output



def ner(model, tokenizer, text):

    tok_ids = tokenizer.tokenize(text)
    with torch.no_grad():
        input = torch.tensor(tok_ids).unsqueeze(0)
        entities_tags = model(input, attention_mask=torch.ones(input.shape))
    
    entities_tags = entities_tags.squeeze(0).tolist()
    print(tok_ids)
    print(entities_tags)
    print([_LABEL_TO_TARGET[e_tag] for e_tag in entities_tags])
    
    # transform tag to type
    entities_types = []
    for e_tag in entities_tags:
        entities_types.append(_LABEL_TO_TARGET[e_tag])

    # create an array of objects of the type {'type': PERSON, 'value': mario rossi, 'offset': 0}
    entities_list = []
    prev = 'O'
    curr_ids = []
    curr_dict = {}
    for count, (id, type) in enumerate(zip(tok_ids, entities_types)):
        #pdb.set_trace()
        if type == 'O' and prev != 'O':
            #pdb.set_trace()
            curr_dict = {'type': _SHORT_TO_TYPE[prev], 'value': tokenizer.detokenize(curr_ids), 'offset': offset}
            entities_list.append(curr_dict)
            curr_ids = []
            prev = 'O'
        elif type[2:] == prev and type != 'O':
            curr_ids.append(id)
        elif type[0] == 'B' or (type[0] == 'I' and type[2:] != prev):
            if len(curr_ids) != 0:
                #pdb.set_trace()
                curr_dict = {'type': _SHORT_TO_TYPE[prev], 'value': tokenizer.detokenize(curr_ids), 'offset': offset}
                entities_list.append(curr_dict)
                curr_ids = []
            curr_ids.append(id)
            #computes the offset for the current entity excluding the <s>
            offset = len(tokenizer.detokenize(tok_ids[1:count]))
            # take into account the space contained in the next token
            if offset > 0:
                offset += 1      
            prev = type[2:]

    return entities_list









if __name__ == '__main__':
    out = main(sys.argv[1:])
    print(out)
