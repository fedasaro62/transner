import math
import os
import pdb
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import Scorer
from config import SetupParameters, TrainingParameters
from NERmodel import BertNER
from wikiNER import WikiNER

__DEBUG = False
__FREEZE_BERT = False



def train(train_set, val_set, device, params, criterion):
        
        # creates the directory for the checkpoints
        os.makedirs(os.path.dirname(SetupParameters.SAVING_PATH), exist_ok=True)

        # initialize the loader
        train_generator = DataLoader(train_set, **params)

        model = BertNER()
        if __FREEZE_BERT:
                # freeze bert layers
                for count, param in enumerate(model.bert.parameters()):
                        param.requires_grad = False
        model.to(device)
        
        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=TrainingParameters.LEARNING_RATE, weight_decay=0.1)
        optimizer = torch.optim.Adam([
                        {"params": model.bert.parameters(), "lr": TrainingParameters.BERT_LEARNING_RATE},
                        {"params": model.cls_layer.parameters(), "lr": TrainingParameters.LINEAR_L_LEARNING_RATE}
                        ],
                        weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = TrainingParameters.MILESTONES, gamma = TrainingParameters.GAMMA)

        # contains the value losses for each epoch
        train_losses = []
        val_losses = []
        min_loss = math.inf
        # training loop
        for epoch in range(TrainingParameters.EPOCHS_NUM):
                model.train()
                train_epoch_losses = []
                for step, (raw_sentences, local_batch, local_labels, local_attention_mask) in enumerate(train_generator):
                        #pdb.set_trace()
                        curr_batch_dim = local_batch.shape[0]
                        # INPUT SIZE = B x 512
                        # LABEL SIZE = B x 512
                        # ATTENTION MASK = B x 512
                        input = local_batch.to(device)
                        labels = local_labels.to(device)
                        attention_mask = local_attention_mask.to(device)
                        if __DEBUG:
                                logits = torch.rand((input.shape[0], SetupParameters.BERT_INPUT_LIMIT, 9))
                        else:
                                logits, _ = model(input, attention_mask)
                        
                        #loss accept only 2D logits, so unpack and repack
                        
                        losses_list = torch.zeros(curr_batch_dim).to(device)
                        tot_loss = 0
                        for batch_idx in range(curr_batch_dim):
                                curr_loss = criterion(logits[batch_idx], labels[batch_idx])
                                losses_list[batch_idx] = curr_loss.item()
                                tot_loss += curr_loss
                        #pdb.set_trace()
                        tot_loss /= curr_batch_dim
                        
                        tot_loss.backward()
                        curr_mean = np.mean(np.array(losses_list.to('cpu')))
                        train_epoch_losses.append(curr_mean)

                        optimizer.step()
                        optimizer.zero_grad()
                        

                val_epoch_losses, preds, golds = infer(model, val_set, device, criterion)


                #LOSSES COMPUTATION
                train_epoch_mean_loss = np.mean(np.array(train_epoch_losses))
                train_losses.append(train_epoch_mean_loss)

                val_epoch_mean_loss = np.mean(np.array(val_epoch_losses))
                val_losses.append(val_epoch_mean_loss)

                #SCORES COMPUTATION
                _, _, f1 = Scorer.score(golds, preds, average='micro')


                print('EPOCH #'+str(epoch)+' :: train loss='+str(train_epoch_mean_loss)+' | val loss='+str(val_epoch_mean_loss)+', f1='+str("%.2f" % f1))


                if val_epoch_mean_loss < min_loss:
                        min_loss = val_epoch_mean_loss
                        torch.save(model.cpu().state_dict(),\
                                SetupParameters.SAVING_PATH+'state_dict.pt')
                        model.to(device)

                scheduler.step()

        return model




def infer(model, test_set, device, criterion):

        test_generator = DataLoader(test_set, **params)

        model.to(device)
        losses = []
        predictions_l = []
        gold_targets = []

        with torch.no_grad():
                model.eval()
                for (raw_sentences, local_batch, local_labels, local_attention_mask) in test_generator:
                        curr_batch_dim = local_batch.shape[0]
                        input = local_batch.to(device)
                        labels = local_labels.to(device)
                        attention_mask = local_attention_mask.to(device)
                        
                        if __DEBUG:
                                logits = torch.rand((TrainingParameters.BATCH_SIZE, SetupParameters.BERT_INPUT_LIMIT, 9))
                        else:
                                logits, prediction = model(input, attention_mask)

                        #loss accepts only 2D logits, so unpack and repack
                        loss = torch.zeros(curr_batch_dim).to(device)
                        for batch_idx in range(curr_batch_dim):
                                loss[batch_idx] = criterion(logits[batch_idx], labels[batch_idx]).item()
                        curr_mean = np.mean(np.array(loss.to('cpu')))
                        losses.append(curr_mean)

                        # accuracy computation
                        final_prediction = torch.argmax(prediction, dim=-1)
                        # put the labels into a one dimensional list
                        predictions_l.extend(final_prediction.view(-1).tolist())
                        gold_targets.extend(labels.view(-1).tolist())


        return losses, predictions_l, gold_targets


def adjust_sizes(train_size, val_size, test_size):
        if train_size + val_size + test_size < dataset.__len__():
                offset = adjust_sizes(train_size+1, val_size, test_size)
        else:
                return 0
        return offset + 1


if __name__ == '__main__':
        start = time.time()

        # CUDA for PyTorch
        if not __DEBUG:
                use_cuda = torch.cuda.is_available()
                device = torch.device("cuda:0" if use_cuda else "cpu")
                if use_cuda:
                        device = torch.cuda.current_device()
                        device_name = torch.cuda.get_device_name(device)
                        print('gpu name = '+device_name)

        else:
                device = 'cpu'
        print('active device = '+str(device))
        

        file_path = r'./wikinerIT'
        dataset = WikiNER(file_path)
        print('Dataset len: '+str(dataset.__len__()))

        # print the types of entities
        types = dataset.get_labels()
        print(types)
        #print the max length of an article
        max_len = dataset.get_max_sentence_len(dataset.data)
        print('max sentence length: '+str(max_len))
        
        

        # create train, validation and test sets
        train_size = int(np.floor(TrainingParameters.DATASET_SPLIT[0] * dataset.__len__()))
        val_size = int(np.floor(TrainingParameters.DATASET_SPLIT[1] * dataset.__len__()))
        test_size = int(np.floor(TrainingParameters.DATASET_SPLIT[2] * dataset.__len__()))
        # it can happen that the dataset is not perfectly divisible
        train_size += adjust_sizes(train_size, val_size, test_size)
        
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        params = {'batch_size': TrainingParameters.BATCH_SIZE,
                'shuffle': True,
                'num_workers': TrainingParameters.WORKERS_NUM}
        criterion = torch.nn.CrossEntropyLoss().to(device)
        
        model = train(train_set, val_set, device, params, criterion)

        _, preds, golds = infer(model, test_set, device, criterion)
        _, _, f1 = Scorer.score(golds, preds, average='micro')
        print('test f1 score= '+str("%.2f" % f1))


        end = time.time()
        h_count = (end-start)/60/60
        print('training time: '+str(h_count)+'h')
