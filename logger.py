from os import write
import json


class CustomLogger:

    def __init__(self, filename):
        self.logfile = open(filename, 'a+')

    def log(self, log_dict):
        self.logfile.write(json.dumps(log_dict)+'\n')