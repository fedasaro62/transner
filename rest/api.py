import getopt
import json
import logging
import os
import pdb
import sys
import time

from flask import Flask, abort, jsonify, make_response, request
from flask_cors import CORS
from simpletransformers.ner.ner_model import NERModel

from config import SetupParameters

app = Flask(__name__)
cors = CORS(app)

os.makedirs('log/', exist_ok=True)
logging.basicConfig(level=logging.INFO, filename='log/ner.log')
model_dict = {'NERmodel': None}


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



# curl -i -H "Content-Type: application/json" -X POST -d '{"strings": ["Vincenzo G. Fonzi è nato a Caserta il 13/08/1983", "Il seguente documento è firmato in calce per il signor Di Marzio.", "Conferma di avvenuto pagamento a Poste Italiane da parte del sig. Giuseppe Maria Boccardi."]}' http://localhost:5000/ner_api/v0.1/ner
# curl -i -H "Content-Type: application/json" -X POST -d '{"strings": ["Maria Santos è nata a Cardenas il 13/08/1983", "The following documents were signed by John Stewart at Berlin headquarters of Deutsche Bank", "Bevestiging van betaling aan ABN AMRO door dhr. Rutger Verhoeven."]}' http://localhost:5000/ner_api/v0.1/ner

@app.route('/ner_api/v0.1/ner', methods=['POST'])
def ner():
    start = time.time()
    input_strings = request.get_json()['strings']
    model = model_dict['NERmodel']

    #model = NERModel('bert', SetupParameters.ITA_MODEL, args={'no_cache': True, 'use_cached_eval_features': False})
    predictions, _ = model.predict(input_strings)

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
    end = time.time()
    execution_t = end-start

    logging.info("-----\nUser ip: {}\nTimestamp: {}\nResponse time: {:5.2f}s\nInput strings: {}\nResponse: {}\n-----".format(request.remote_addr, start, execution_t, input_strings, results))

    return jsonify(results), 200



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':

    model_dict['NERmodel'] = NERModel('bert', SetupParameters.IT_MODEL, use_cuda= False, args={'no_cache': True, 'use_cached_eval_features': False, 'process_count': 1, 'silent': True})
    app.run(host='0.0.0.0', debug=True, port=5000)
