import getopt
import json
import logging
import os
import pdb
import sys
import time

from utils.preprocessing import NERSeparatePunctuations
from utils.ner import make_ner_dict, find_from_regex
from flask import Flask, abort, jsonify, make_response, request
from flask_cors import CORS
from simpletransformers.ner.ner_model import NERModel

from config import SetupParameters

app = Flask(__name__)
cors = CORS(app)

os.makedirs('log/', exist_ok=True)
logging.basicConfig(level=logging.INFO, filename='log/ner.log')
model_dict = {'NERmodel': None}




@app.route('/transner/v0.3/ner', methods=['POST'])
def ner():
    """This API returns the extracted entities for each sentence.
    
    Returns:
        [list] -- list of dictionaries containing, for each input string, the list of entities with type, value and offset.
        e.g.
        {'sentence': 'Mr. Robinson lives in Reeds.'
        'entities':[
            {
                'offset': 0,
                'type': PER,
                'value': Mr. Robinson
            },
            {
                'offset': 22,
                'type': LOC,
                'value': Reeds 
            }
        ]} 
    """
    start = time.time()

    raw_input_strings = request.get_json()['strings']

    model = model_dict['NERmodel']

    # define the preprocesser to use and apply it
    preprocesser = NERSeparatePunctuations()
    input_strings = preprocesser.preprocess(raw_input_strings, do_lower_case=True)

    # use the trained model to extract the PER, LOC, ORG, MISC entity types
    predictions, _ = model.predict(input_strings)
    assert len(predictions) == len(input_strings)

    # retrieve the dictionary with with sentences and entities 
    ner_dict = make_ner_dict(input_strings, predictions)

    # get original strings (no preprocessed) and adjust the entities offset
    preprocesser.adjustEntitiesOffset([r['entities'] for r in ner_dict], adjust_case=True)
    for r, original in zip(ner_dict, raw_input_strings):
        r['sentence'] = original

    # extract other entities by using regex
    ner_dict = find_from_regex(ner_dict)

    end = time.time()
    execution_t = end-start

    logging.info("-----\nUser ip: {}\nTimestamp: {}\nResponse time: {:5.2f}s\nInput strings: {}\nResponse: {}\n-----"
                        .format(request.remote_addr, start, execution_t, input_strings, ner_dict))

    return jsonify(ner_dict), 200



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':

    model_dict['NERmodel'] = NERModel('bert', SetupParameters.PRETRAINED_MODEL, use_cuda=False, args={'no_cache': True, 'use_cached_eval_features': False, 'process_count': 1, 'silent': True})
    app.run(host='0.0.0.0', debug=True, port=5000)
