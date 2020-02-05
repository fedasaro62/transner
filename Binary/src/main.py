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
    
    # transform tag to type
    entities_types = []
    for e_tag in entities_tags:
        entities_types.append(_LABEL_TO_TARGET[e_tag])
    
    """
    for id, type in zip(tok_ids, entities_types):
        if type == 'O':
            continue
        else:
            print(tokenizer.detokenize(id)+' : '+type)
    """
    
    """
    prev = 'O'
    curr_ids = []
    for id, type in zip(tok_ids, entities_types):
        #pdb.set_trace()
        if type == 'O' and prev != 'O':
            print(tokenizer.detokenize(curr_ids)+' : '+prev)
            curr_ids = []
            prev = 'O'
        elif type[2:] == prev and type != 'O':
            curr_ids.append(id)
        elif type[0] == 'B' or (type[0] == 'I' and type[2:] != prev):
            if len(curr_ids) != 0:
                print(tokenizer.detokenize(curr_ids)+' : '+prev)
                curr_ids = []
            curr_ids.append(id)
            prev = type[2:]
    """

    e_list = []
    prev = 'O'
    curr_ids = []
    curr_dict = dict()
    for count, (id, type) in enumerate(zip(tok_ids, entities_types)):
        #pdb.set_trace()
        if type == 'O' and prev != 'O':
            #pdb.set_trace()
            curr_dict = {'type': prev, 'value': tokenizer.detokenize(curr_ids), 'offset': offset}
            e_list.append(curr_dict)
            curr_ids = []
            prev = 'O'
        elif type[2:] == prev and type != 'O':
            curr_ids.append(id)
        elif type[0] == 'B' or (type[0] == 'I' and type[2:] != prev):
            if len(curr_ids) != 0:
                #pdb.set_trace()
                curr_dict = {'type': prev, 'value': tokenizer.detokenize(curr_ids), 'offset': offset}
                e_list.append(curr_dict)
                curr_ids = []
            curr_ids.append(id)
            #computes the offset for the current entity excluding the <s>
            offset = len(tokenizer.detokenize(tok_ids[1:count])) #take into account the space
            if tokenizer.detokenize(tok_ids[count+1])[0] == ' ':
                offset += 1
            
            
            prev = type[2:]

    print(e_list)







if __name__ == '__main__':
    main(sys.argv[1:])
