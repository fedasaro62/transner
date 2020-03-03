import getopt
import json
import pdb
import sys
import time

from simpletransformers.ner.ner_model import NERModel
from flask import Flask, abort, jsonify, make_response, request

from config import SetupParameters


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



# curl -i -H "Content-Type: application/json" -X POST -d '{"strings": ["Vincenzo G. Fonzi è nato a Caserta il 13/08/1983", "Il seguente documento è firmato in calce per il signor Di Marzio.", "Conferma di avvenuto pagamento a Poste Italiane da parte del Sig. Giuseppe Maria Boccardi."]}' http://localhost:5000/ner_api/v0.1/ner

@app.route('/ner_api/v0.1/ner', methods=['POST'])
def ner():
    input_strings = request.get_json()['strings']
    model = NERModel('bert', SetupParameters.ITA_MODEL, args={'no_cache': True, 'use_cached_eval_features': False})
    predictions, _ = model.predict(input_strings)
    #pdb.set_trace()

    assert len(predictions) == len(input_strings)
    results = []
    curr_res = {}
    for s, prediction in zip(input_strings, predictions):
        curr_res = dict()
        curr_res['sentence'] = s
        curr_res['entities'] = []
        curr_offset = 0
        #for multi-word entities
        beginning_offset = None
        active_e_type = None
        active_e_value = ''

        for e_pred in prediction:
            kv_pair = list(e_pred.items())
            assert len(kv_pair) == 1
            #pdb.set_trace()
            e_value, e_type = kv_pair[0]
            if e_type[0] == 'B':
                #if a entity is still active, close it
                if active_e_type:
                    curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 'value': active_e_value[:-1], 'offset': beginning_offset}
                    curr_res['entities'].append(curr_entity)
                beginning_offset = curr_offset
                active_e_type= e_type[2:]
                active_e_value += e_value + ' '
            elif e_type[0] == 'I':
                #treat it as a beginner if the beginner is not present
                if not active_e_type:
                    beginning_offset = curr_offset
                    active_e_type= e_type[2:]
                    active_e_value += e_value + ' '
                elif e_type[2:] == active_e_type:
                    active_e_value += e_value + ' '
                else:
                    curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 'value': active_e_value[:-1], 'offset': beginning_offset}
                    curr_res['entities'].append(curr_entity)
                    beginning_offset = curr_offset
                    active_e_type= e_type[2:]
                    active_e_value += e_value + ' '
            elif e_type[0] == 'O' and active_e_type:
                curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 'value': active_e_value[:-1], 'offset': beginning_offset}
                curr_res['entities'].append(curr_entity)
                beginning_offset = None
                active_e_type = None
                active_e_value = ''

            #offset takes into account also the space
            curr_offset += len(e_value) + 1
            #if last prediction for that string, then save the active entities
            if curr_offset > len(s) and active_e_type:
                curr_entity = {'type': _SHORT_TO_TYPE[active_e_type], 'value': active_e_value[:-1], 'offset': beginning_offset}
                curr_res['entities'].append(curr_entity)
        results.append(curr_res)

    return jsonify(results)



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
    
    

import pdb
if __name__ == '__main__':
    #pdb.set_trace()
    app.run(debug=True, port=5000)


    """
def ner():
    
    #input in the format {'strings': ['string1', 'string2', ...]}
    

    input_strings = request.get_json()['strings']
    dictfile = SetupParameters.LOAD_PATH

    # load the model and the tokenizer
    model = BertNER(SetupParameters.MODEL_SEED)
    state_dict = torch.load(dictfile)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = NERTokenizer(SetupParameters.TOKENIZER_ID)
    
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
"""
