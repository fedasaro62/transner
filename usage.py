import pdb
import argparse
from transner import Transner



def main(args):
    model = Transner(pretrained_model='bert_uncased_base_easyrights_v0.1',
                    use_cuda=args.cuda,
                    n_gpus=args.n_gpus,
                    quantization=True)
    ner_dict = model.ner(args.strings, apply_regex=True, apply_gazetteers=True)
    #ner_dict = model.find_dates(ner_dict)
    print(ner_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--strings', 
        nargs='+', 
        help='List of strings for the NER', 
        required=True)
    parser.add_argument(
        '--cuda',
        action='store_true', 
        required=False,
        default=False, 
        help='Flag to use GPU (default: False)')
    parser.add_argument(
        '--n_gpus',
        type=int,
        required=False,
        default=1,
        help='Number of gpus to use if cuda flag was set (default: 1)')
    args = parser.parse_args()

    main(args)
    
    