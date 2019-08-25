from __future__ import unicode_literals, print_function

import os
import spacy
import logging
from argparse import ArgumentParser
from recruitment_agency_detector.data_loader import get_data_with_details
from recruitment_agency_detector.config import load_config
from recruitment_agency_detector import set_logging_level, LOGGER

def process_batch(classifier, test_dir, output_file):
    fh_output = open(output_file, 'w')
    fh_output.write('id\torg_name\tsite\tnew_predict\told_predict\turl\tscore\n')
    for test_text, category, id, orgname, site, url in get_data_with_details(test_dir):
        doc = classifier(test_text)
        predict_cat = 'yes' if doc.cats['yes'] > doc.cats['no'] else 'no'
        fh_output.write(f"{id}\t{orgname}\t{site}\t{predict_cat}\t{category}\t{url}\t{doc.cats}\n")
    fh_output.close()


def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='''
                            process a test batch using a trained model
                            ''')
    parser.add_argument('config', help='training config file', type=str)
    parser.add_argument('--test_set', help='test set', type=str)
    parser.add_argument('--output_dir', help='test set', type=str,
            default='res')

    return parser.parse_args()


def main(args):
    config = load_config(args.config)

    classifier = spacy.load(config['model_path'])
    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_set:
        test_sets = args.test_set.split(",")
    else:
        test_sets = config['datasets']['test']

    for data_set in test_sets:
        output_file = os.path.join(
            args.output_dir,
            data_set + '.tsv')
        LOGGER.info('process %s and save result to %s', data_set, output_file)
        process_batch(
            classifier,
            config['datasets']['test'][data_set],
            output_file
        )


if __name__ == "__main__":
    set_logging_level(logging.INFO)
    LOGGER.info("starting...")
    args = get_args()

    main(args)
