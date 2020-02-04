import getopt
import pdb
import sys

import torch

from model import BertNER
from tokenizer import NERTokenizer


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
    print(entities_tags.squeeze(0))









if __name__ == '__main__':
    main(sys.argv[1:])
