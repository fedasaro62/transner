import pdb
import argparse
from transner import Transner



def main(strings):
    model = Transner(pretrained_model='multilang_uncased', use_cuda=False)
    ner_dict = model.ner(strings, apply_regex=True)
    print(ner_dict)





if __name__ == '__main__':
    """Input example:

        $python usage.py --strings \
            "Mario è nato a Milano" \
            "Nicola è andato a Firenze"
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s',
        '--strings', 
        nargs='+', 
        help='List of strings for the NER', 
        required=True)
    args = parser.parse_args()

    main(args.strings)
    
    