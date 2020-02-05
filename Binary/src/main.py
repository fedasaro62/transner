import getopt
import pdb
import sys

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



"""
main.py -f <state_dict_file> -s <string>
"""

def main(argv):
 
    dictfile = ''
    input_string = ''
    try:
        opts, args = getopt.getopt(argv,"f:s:",["dictfile=","string="])
    except getopt.GetoptError:
        print ('main.py -f <state_dict_file> -s <string>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('main.py -f <state_dict_file> -s <string>')
            sys.exit()
        elif opt in ("-f", "--dictfile"):
            dictfile = arg
        elif opt in ("-s", "--string"):
            input_string = arg
        else:
            print('-h for help')
            sys.exit(2)
    if dictfile == '' or input_string == '':
        print('Missing argument (-h for help)')
        sys.exit(2)
    
    ner(state_dict=torch.load(dictfile), text=input_string)


def ner(state_dict, text):

    model = BertNER()
    state_dict
    model.load_state_dict(state_dict)
    
    tokenizer = NERTokenizer()
    tok_ids = tokenizer.tokenize(text)
    
    model.eval()
    with torch.no_grad():
        input = torch.tensor(tok_ids).unsqueeze(0)
        entities_tags = model(input, attention_mask=torch.ones(input.shape))
    
    entities_tags = entities_tags.squeeze(0).tolist()
    print(tok_ids)
    print(entities_tags)
    print([_LABEL_TO_TARGET[e_tag] for e_tag in entities_tags])
    """
    # transform tag to type
    entities_types = []
    for e_tag in entities_tags:
        entities_types.append(_LABEL_TO_TARGET[e_tag])
    """
    
    
    """
    prev = ''
    curr_ids = []
    for id, type in zip(tok_ids, entities_types):
        pdb.set_trace()
        if type != prev and prev != '':
            if len(curr_ids) != 0:
                pdb.set_trace()
                print(tokenizer.detokenize(curr_ids)+' : '+prev)
                curr_ids = []

        if type == 'O':
            prev = 'O'
            continue

        elif type[2:] == prev:
            curr_ids.append(id)

        else:
            curr_ids.append(id)
            prev = type[2:]

    if len(curr_ids) != 0:
        print(tokenizer.detokenize(curr_ids)+' : '+prevs)
    


    word = tokenizer.detokenize(id)
    e_type = _LABEL_TO_TARGET[tag]
    print(word+' : '+e_type)
    """









if __name__ == '__main__':
    main(sys.argv[1:])
