from __future__ import unicode_literals, print_function

import spacy
from argparse import ArgumentParser
import logging
from recruitment_agency_detector.model import Model
from recruitment_agency_detector.config import load_config
from recruitment_agency_detector import set_logging_level, LOGGER

def process_batch(classifier, test_dir, output_file):
    fh_output = open(output_file, 'w')
    fh_output.write('id\torg_name\tsite\tnew_predict\told_predict\turl\tscore\n')
    for test_text, category, id, orgname, site, url in get_data_with_details(test_dir):
        result = model.process_with_saved_model(test_text)
        predict_cat = 'yes' if result[0] > result[1] else 'no'
        fh_output.write(f"{id}\t{orgname}\t{site}\t{predict_cat}\t{category}\t{url}\t{result}\n")
    fh_output.close()


def train(config):
    config = load_config(config)
    config['action'] = 'train'
    model = Model(config)
    model.build_and_train()
    if model.type.startswith('spacy'):
        model.save(config['model_path'])


def predict(args):
    config = load_config(args.config)
    config['action'] = 'inference'
    model = Model(config)
    model.load()

    if args.test_set:
        test_sets = args.test_set.split(",")
    else:
        test_sets = config['datasets']['test']

    os.makedirs(args.output_dir, exist_ok=True)

    for data_set in test_sets:
        output_file = os.path.join(
            args.output_dir,
            data_set + '.tsv')
        LOGGER.info('process %s and save result to %s', data_set, output_file)
        process_batch(
            model,
            config['datasets']['test'][data_set],
            output_file
        )



def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='''
                            train a model
                            ''')
    parser.add_argument('config', help='training config file', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    set_logging_level(logging.INFO)
    LOGGER.info("starting...")
    args = get_args()
    predict(args.config)
