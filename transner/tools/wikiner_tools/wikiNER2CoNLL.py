import pdb
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from dataset_wikiNER import WikiNER


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




if __name__ == '__main__':
    """Example
        python wikiNER2CoNLL.py --infile wikiner_original/aijwikineritwp2\
                                --outfile wikiner_conll/it/wikinerIT.conll\
                                --trainsplit .7\
                                --valsplit .15\
                                --testsplit .15\
                                --augment

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(        
        "--infile",
        default=None,
        type=str,
        required=True,
        help="Path to input file")
    parser.add_argument(        
        "--outfile",
        default=None,
        type=str,
        required=True,
        help="Path to output file")
    parser.add_argument(        
        "--valsplit",
        default=None,
        type=float,
        required=True,
        help="Split ratio for validation")
    parser.add_argument(        
        "--trainsplit",
        default=None,
        type=float,
        required=True,
        help="Split ratio for training")
    parser.add_argument(        
        "--testsplit",
        default=None,
        type=float,
        required=True,
        help="Split ratio for test")
    parser.add_argument(        
        "--randsplit",
        default=False,
        action='store_true',
        required=False,
        help="To apply random split")
    parser.add_argument(        
        "--augment",
        default=False,
        action='store_true',
        required=False,
        help="To apply data augmentation")

    args = parser.parse_args()

    if len(args.infile) == 0 or len(args.outfile) == 0:
        print('Missing arguments')
        exit(2)
    if args.valsplit < 0 or args.valsplit > 1 or args.trainsplit < 0 or args.trainsplit > 1 or args.testsplit < 0 or args.testsplit > 1:
        print('The training, validation and test factors must in the range [0,1]')
        exit(2)
    if not (args.valsplit + args.testsplit + args.trainsplit == 0 or args.valsplit + args.testsplit + args.trainsplit == 1):
        print('The sum of training, validation and test factors must be 1')
        exit(2)

    dataset = WikiNER(args.infile)
    
    if args.valsplit + args.testsplit + args.trainsplit == 1:
        # create train, validation and test sets
        train_size = int(np.floor(args.trainsplit * dataset.__len__()))
        val_size = int(np.floor(args.valsplit * dataset.__len__()))
        test_size = int(np.floor(args.testsplit * dataset.__len__()))
        # it can happen that the dataset is not perfectly divisible
        train_size += adjust_sizes(train_size, val_size, test_size)
        
        if args.randsplit:
            train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        else:
            indices = list(range(len(dataset)))
            train_set = Subset(dataset, indices[:train_size])
            val_set = Subset(dataset, indices[train_size: train_size+val_size])
            test_set = Subset(dataset, indices[train_size+val_size:])

        wikiNER2CoNLL(train_set, args.outfile+'.train', augment=args.augment)
        wikiNER2CoNLL(val_set, args.outfile+'.val', augment=args.augment)
        wikiNER2CoNLL(test_set, args.outfile+'.test', augment=args.augment)

        print('Output files: '+args.outfile.split('/')[-1]+'.train'+', '+args.outfile.split('/')[-1]+'.val'+', '+args.outfile.split('/')[-1]+'.test')

    else:
        wikiNER2CoNLL(dataset, args.outfile, augment=args.augment)
        print('Output file: '+args.outfile.split('/')[-1])
