from __future__ import unicode_literals, print_function

import spacy
import random
from argparse import ArgumentParser
from xml_miner.miner import TRXMLMiner
from .model import Model
from .config import load_config
from . import LOGGER
from .spacy_model import load_data

def main(config):
    config = load_config(config)
    model = Model(config)
    model.build_graph()
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(config)
    model.train(train_texts, train_cats, dev_texts, dev_cats)
    model.save_spacy_model(config['model_path'])

def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='''
                            the skill validation model
                            ''')
    parser.add_argument('config', help='training config file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)

    #plac.call(evaluate_and_print)
    main(config)

    #plac.call(predict)
