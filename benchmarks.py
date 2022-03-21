import argparse
import json
import logging
import os
import pdb
import sys
import time

import requests
from flask import Flask, abort, jsonify, make_response, request
from flask_cors import CORS
from memory_profiler import memory_usage, profile

from config import SetupParameters
from transner import Transner

app = Flask(__name__)
cors = CORS(app)

model_dict = {'NERmodel': None}


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

_USE_CUDA=True
_N_GPUS = 3
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in range(_N_GPUS)])
_USE_QUANTIZE=False
_BATCH_SIZE=8


@app.route('/transner/v0.7/benchmarks', methods=['POST'])
def benchmarks():
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
    device = 'GPU' if _USE_CUDA else 'CPU'
    test_string = request.get_json()['strings'][0]
    sentence_len = len(test_string.split())
    test_array = [test_string for _ in range(_BATCH_SIZE)]
    print('device={}; batch={}; sentence_len={}'.format(device, _BATCH_SIZE, sentence_len))

    exec_times = []
    mem_usages = []
    for iteration in range(10):
        print('------- RUN #{} -------'.format(iteration+1))
        start = time.time()
        curr_mem_usage = memory_usage((do_ner, (test_array,)))
        end = time.time()
        mem_usages.append(max(curr_mem_usage))
        exec_times.append(end-start)
    max_mem = max(mem_usages)
    min_mem = min(mem_usages)
    avg_mem = sum(mem_usages)/len(mem_usages)
    avg_ms = sum(exec_times) / len(exec_times)
    m_count = (avg_ms/60) % 60
    s_count = avg_ms % 60
    ms_count = avg_ms - int(avg_ms)
    mem_str='max_mem={} MiB; min_mem={} MiB; avg_mem={} MiB'.format(round(max_mem), round(min_mem), round(avg_mem))
    time_str='execution time: {}m:{}s:{}ms'.format(round(m_count), round(s_count), int(ms_count*1000))
    print(mem_str)
    print(time_str)
    print('---------------------')

    return jsonify('{}    {}'.format(mem_str, time_str)), 200


@profile
def do_ner(raw_input_strings):
    # select the model and run NER
    model = model_dict['NERmodel']
    # reset preprocesser internal structures
    model.reset_preprocesser()
    ner_dict = model.ner(raw_input_strings, apply_regex=True, apply_gazetteers=True)
    return ner_dict



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        required=True,
        help='Port to allocate for the rest api'
    )
    args = parser.parse_args()

    model_dict['NERmodel'] = Transner(pretrained_model=SetupParameters['PRETRAINED_MODEL'],
                                    quantization=_USE_QUANTIZE,
                                    use_cuda=_USE_CUDA,
                                    n_gpus=_N_GPUS)
    app.run(host='0.0.0.0', debug=True, port=args.port)
