import pdb

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoModel, CamembertConfig

from config import SetupParameters, TrainingParameters


class BertNER(nn.Module):


    def __init__(self):

        super(BertNER, self).__init__()
        torch.manual_seed(TrainingParameters.SEED)

        config = CamembertConfig.from_pretrained(SetupParameters.MODEL_ID, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(SetupParameters.MODEL_ID, config=config)
        
        self.cls_layer = nn.Sequential(nn.Linear(768*4, 9),
                                        nn.ReLU())
        self._softmax = F.softmax


    def forward(self, input, attention_mask):
        
        # segment composed only of 1 sentence
        out = self.bert(input, attention_mask = attention_mask)
        #cntx_emb = out[0]

        #tuple of 4xBxLENx768
        cntx_emb = out[2][-4:]
        #concat tensors along the last dimension. Size is then BxLENx3072
        concat_emb = torch.cat(cntx_emb, dim=-1)
        logits = self.cls_layer(concat_emb)
        prediction = self._softmax(logits, dim=-1)
        
        return logits, prediction


    @staticmethod
    def get_model_trainable_params(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
