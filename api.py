import argparse
import json
import os
import pdb
import time
from logger import CustomLogger
import logging
from logstash_formatter import LogstashFormatterV1

import requests
from flask import Flask, abort, jsonify, make_response, request
from flask_cors import CORS

from config import SetupParameters
from transner import Transner

app     = Flask(__name__)
cors    = CORS(app)

os.makedirs('log/', exist_ok=True)

flasklog    = open('log/flask.log', 'a+')
handler     = logging.StreamHandler(stream=flasklog)
handler.setFormatter(LogstashFormatterV1())
logging.basicConfig(handlers=[handler], level=logging.INFO)

_MAX_LEN = 150


@app.route('/transner/v0.7/ner', methods=['POST'])
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
    start               = time.time()
    raw_input_strings   = request.get_json()['strings']
    for s in raw_input_strings:
        if len(s.split()) > _MAX_LEN:
            abort(400)
    model               = app.config['NERmodel']
    # reset preprocesser internal structures
    model.reset_preprocesser()
    ner_dict            = model.ner(raw_input_strings, apply_regex=True, apply_gazetteers=True)

    end                 = time.time()
    execution_t         = end-start

    log_dict = {'remote_addr': request.remote_addr,
                'start': start,
                'response_time': execution_t,
                'request': raw_input_strings,
                'response': ner_dict}
    app.config['logger'].log(log_dict)

    return jsonify(ner_dict), 200


@app.route('/transner/v0.7/supported_types', methods=['GET'])
def supported_types():
    with open('supported_types.json') as fp:
        return jsonify(json.load(fp))


@app.route('/transner/v0.7/max_len', methods=['GET'])
def get_max_len():
    return jsonify({'max_len': _MAX_LEN})


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Size too large. Max size {} words'.format(_MAX_LEN)}), 400)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        required=True,
        help='Port to allocate for the rest api'
    )
    parser.add_argument(
        '--cuda',
        required=False,
        default=False,
        action='store_true',
        help='Flag to enable inference on gpu'        
    )
    parser.add_argument(
        '--multi_gpu',
        type=bool,
        required=False,
        default=False,
        help='Flag to user multiple gpus (default False if cuda=True). To set the visible gpus please set the enviroment variables \
            CUDA_DEVICE_ORDER and CUDA_VISIBLE_DEVICES (see reference on NVIDIA documentation)'
    )
    args = parser.parse_args()
    app.config['NERmodel'] = Transner(pretrained_model=SetupParameters['PRETRAINED_MODEL'], 
                                        use_cuda=args.cuda, 
                                        multi_gpu=args.multi_gpu,
                                        threshold=0.75)
    app.config['logger'] = CustomLogger('log/ner.log')
    app.run(host='0.0.0.0', debug=False, port=args.port)
