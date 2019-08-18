from __future__ import unicode_literals, print_function

import spacy
from argparse import ArgumentParser
from recruitment_agency_detector.model import Model
from recruitment_agency_detector.config import load_config

def main(config):
    config = load_config(config)
    config['action'] = 'train'
    model = Model(config)
    model.build_and_train()
    model.save(config['model_path'])

def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='''
                            the skill validation model
                            ''')
    parser.add_argument('config', help='training config file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.config)
