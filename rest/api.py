import getopt
import json
import pdb
import sys
import time

import torch
from flask import Flask, abort, jsonify, make_response, request

from ner_model import *


app = Flask(__name__)


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



# curl -i -H "Content-Type: application/json" -X POST -d '{"strings": ["Mario Rossi è nato a Busto Arsizio", "Il signor D'Alberto ha effettuato un pagamento a Matteo", "Marco e Luca sono andati a Magenta"]}' http://localhost:5000/ner_api/v0.1/ner

@app.route('/ner_api/v0.1/ner', methods=['POST'])
def ner():
    """
    input in the format {'strings': ['string1', 'string2', ...]}
    """

    input_strings = request.get_json()['strings']
    dictfile = SetupParameters.LOAD_PATH

    # load the model and the tokenizer
    model = BertNER(SetupParameters.MODEL_ID)
    state_dict = torch.load(dictfile)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = NERTokenizer(SetupParameters.MODEL_ID)
    
    results_l = []
    for s in input_strings:
        entities_list = extract_entities(model, tokenizer, text=s)
        results_l.append({'sentence': s, 'entities': entities_list})
    
    json_output = {'timestamp': time.time(), 'results': results_l}
    #json_output = json.dumps(output, ensure_ascii=False)
    return jsonify(json_output)



def extract_entities(model, tokenizer, text):

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




@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
    
    

import pdb
if __name__ == '__main__':
    #pdb.set_trace()
    app.run(debug=True, port=5000)
