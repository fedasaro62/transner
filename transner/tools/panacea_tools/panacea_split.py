import pdb
import getopt
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from dataset_panacea import Panacea

def wikiNER2CoNLL(dataset, outfile, augment=False):

    with open(outfile, 'w') as file:
        for idx in range(dataset.__len__()):
            (curr_sentence, curr_targets) = dataset.__getitem__(idx)
            if len(curr_sentence.split()) != len(curr_targets):
                raise ValueError('Sentence and target lengths do not match')
            for word, target in zip(curr_sentence.split(), curr_targets):
                file.write(word+' '+target+'\n')
            file.write('\n')
            if augment:
                for word, target in zip(curr_sentence.lower().split(), curr_targets):
                    file.write(word+' '+target+'\n')
                file.write('\n')


def adjust_sizes(train_size, val_size, test_size):
        if train_size + val_size + test_size < dataset.__len__():
                offset = adjust_sizes(train_size+1, val_size, test_size)
        else:
                return 0
        return offset + 1


# python3 panacea_split.py -v .15 -t .7 -e .15 -a -o panacea_conll/panaceaGR.conll -i panacea_original/output_greek.conll

if __name__ == '__main__':

    infile = ''
    outfile = ''
    val_split = 0
    train_split = 0
    test_split = 0
    argv = sys.argv[1:]
    rand_split = False
    augment = False
    try:
        opts, args = getopt.getopt(argv,"hi:o:v:t:e:ra",["help=", "input=", "output=", "validation=", "training=", "test=", "randomsplit="])
    except getopt.GetoptError:
        print ('wikiNER2CoNLL.py -i <input_file> -o <output_file>')
        sys.exit(2)
    for count, (opt, arg) in enumerate(opts):
        if opt in ('-h', '--help'):
            print ('Correct format: wikiNER2CoNLL.py -i <input_file> -o <output_file>')
            sys.exit()
        elif opt in ("-i", "--input"):
            infile = arg
        elif opt in ("-o", "--output"):
            outfile = arg
        elif opt in ("-v", "--validation"):
            val_split = float(arg)
        elif opt in ("-t", "--training"):
            train_split = float(arg)
        elif opt in ("-e", "--test"):
            test_split = float(arg)
        elif opt in ("-r", "--randomsplit"):
            rand_split = True
        elif opt in ("-a", "--augment"):
            augment = True
        else:
            print('Uncorrect format (-h for help)')
            sys.exit(2)
    
    if len(infile) == 0 or len(outfile) == 0:
        print('Missing arguments')
        exit(2)
    if val_split < 0 or val_split > 1 or train_split < 0 or train_split > 1 or test_split < 0 or test_split > 1:
        print('The training, validation and test factors must in the range [0,1]')
        exit(2)
    if not (val_split + test_split + train_split == 0 or val_split + test_split + train_split == 1):
        print('The sum of training, validation and test factors must be 1')
        exit(2)
    #inputfile = r'./wikinerIT'
    #outfile = r'./wikiNER.conll'
    dataset = Panacea(infile)
    print(dataset)
    pdb.set_trace()
    
    if val_split + test_split + train_split == 1:
        # create train, validation and test sets
        train_size = int(np.floor(train_split * dataset.__len__()))
        val_size = int(np.floor(val_split * dataset.__len__()))
        test_size = int(np.floor(test_split * dataset.__len__()))
        # it can happen that the dataset is not perfectly divisible
        train_size += adjust_sizes(train_size, val_size, test_size)
        

        if rand_split:
            train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        else:
            indices = list(range(len(dataset)))
            train_set = Subset(dataset, indices[:train_size])
            val_set = Subset(dataset, indices[train_size: train_size+val_size])
            test_set = Subset(dataset, indices[train_size+val_size:])
            
        wikiNER2CoNLL(train_set, outfile+'.train', augment=augment)
        wikiNER2CoNLL(val_set, outfile+'.val', augment=augment)
        wikiNER2CoNLL(test_set, outfile+'.test', augment=augment)

        print('Output files: '+outfile.split('/')[-1]+'.train'+', '+outfile.split('/')[-1]+'.val'+', '+outfile.split('/')[-1]+'.test')

    else:
        wikiNER2CoNLL(dataset, outfile, augment=augment)
        print('Output file: '+outfile.split('/')[-1])
