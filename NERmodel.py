import pdb

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoModel

from config import SetupParameters, TrainingParameters


class BertNER(nn.Module):


    def __init__(self):

        super(BertNER, self).__init__()
        torch.manual_seed(TrainingParameters.SEED)

        self.bert = AutoModel.from_pretrained(SetupParameters.MODEL_ID)
        self.cls_layer = nn.Sequential(nn.Linear(768, 9),
                                        nn.ReLU())
        self._softmax = F.softmax


    def forward(self, input, attention_mask):
        
        # segment composed only of 1 sentence
        out = self.bert(input, attention_mask = attention_mask)
        cntx_emb = out[0]
        logits = self.cls_layer(cntx_emb)
        prediction = self._softmax(logits, dim=-1)
        
        return logits, prediction


    @staticmethod
    def get_model_trainable_params(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
