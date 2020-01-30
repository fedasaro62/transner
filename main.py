import pdb
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from bertNER import BertNER
from config import SetupParameters, TrainingParameters
from wikiNER import WikiNER



def train(train_set, val_set, device, params, criterion):
        
        # initialize the loader

        train_generator = DataLoader(train_set, **params)

        model = BertNER()
        model.to(device)
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=TrainingParameters.LEARNING_RATE, weight_decay=0.1)

        #infer(model, val_set, device, criterion)
        # contain the value losses for each epoch
        train_losses = []
        val_losses = []
        # training loop
        for epoch in range(TrainingParameters.EPOCHS_NUM):
                model.train()
                train_epoch_losses = []
                for step, (raw_sentences, local_batch, local_labels, local_attention_mask) in enumerate(train_generator):
                        curr_batch_dim = local_batch.shape[0]
                        # INPUT SIZE = B x 512
                        # LABEL SIZE = B x 512
                        # ATTENTION MASK = B x 512
                        input = local_batch.to(device)
                        labels = local_labels.to(device)
                        attention_mask = local_attention_mask.to(device)
                        logits = torch.rand((input.shape[0], SetupParameters.BERT_INPUT_LIMIT, 9))
                        logits, _ = model(input, attention_mask)
                        
                        #loss accept only 2D logits, so unpack and repack
                        
                        loss = torch.zeros(curr_batch_dim).to(device)
                        for batch_idx in range(curr_batch_dim):
                                loss[batch_idx] = criterion(logits[batch_idx], labels[batch_idx]).item()

                        curr_mean = np.mean(np.array(loss.to('cpu')))
                        train_epoch_losses.append(curr_mean)

                        optimizer.step()
                        optimizer.zero_grad()

                val_epoch_losses = infer(model, val_set, device, criterion)


                #LOSSES COMPUTATION
                train_epoch_mean_loss = np.mean(np.array(train_epoch_losses))
                train_losses.append(train_epoch_mean_loss)

                val_epoch_mean_loss = np.mean(np.array(val_epoch_losses))
                val_losses.append(val_epoch_mean_loss)

                print('EPOCH #'+str(epoch)+':: train loss='+str(train_epoch_mean_loss)+' | val loss='+str(val_epoch_mean_loss))

        return model




def infer(model, test_set, device, criterion):

        test_generator = DataLoader(test_set, **params)

        model.to(device)
        losses = []
        with torch.no_grad():
                model.eval()
                for (raw_sentences, local_batch, local_labels, local_attention_mask) in test_generator:
                        curr_batch_dim = local_batch.shape[0]
                        input = local_batch.to(device)
                        labels = local_labels.to(device)
                        attention_mask = local_attention_mask.to(device)
                        
                        #logits = torch.rand((TrainingParameters.BATCH_SIZE, SetupParameters.BERT_INPUT_LIMIT, 9))
                        logits, prediction = model(input, attention_mask)

                        #loss accepts only 2D logits, so unpack and repack
                        loss = torch.zeros(curr_batch_dim).to(device)
                        for batch_idx in range(curr_batch_dim):
                                loss[batch_idx] = criterion(logits[batch_idx], labels[batch_idx]).item()
                        curr_mean = np.mean(np.array(loss.to('cpu')))
                        losses.append(curr_mean)


                        # TODO HERE COMPUTE ALSO ACCURACY (or f1)

        return losses




if __name__ == '__main__':
        start = time.time()

        # CUDA for PyTorch
        
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        print('active device = '+str(device))
        if use_cuda:
                device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device)
                print('gpu name = '+device_name)
        
        #device = 'cpu'


        file_path = r'./wikinerIT'
        dataset = WikiNER(file_path)

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
        
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        params = {'batch_size': TrainingParameters.BATCH_SIZE,
                'shuffle': True,
                'num_workers': 2}
        criterion = torch.nn.CrossEntropyLoss().to(device)

        model = train(train_set, val_set, device, params, criterion)

        infer(model, test_set, device, params, criterion)


        end = time.time()
        h_count = (end-start)/60/60
        print('training time: '+str(h_count)+'h')
