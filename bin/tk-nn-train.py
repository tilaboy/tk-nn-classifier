from __future__ import unicode_literals, print_function
import os
from argparse import ArgumentParser
import logging
from recruitment_agency_detector.model import Model
from recruitment_agency_detector.config import load_config
from recruitment_agency_detector.data_loader import get_data_with_details
from recruitment_agency_detector import set_logging_level, LOGGER

def process_batch(model, test_dir, output_file, config):
    fh_output = open(output_file, 'w')
    fh_output.write('id\torg_name\tsite\tnew_predict\told_predict\turl\tscore\n')
    for test_text, category, id, orgname, site, url in get_data_with_details(test_dir, config):
        probabilities = model.process_with_saved_model(test_text)
        # todo: this is the index of classes, still need to map back
        predict_cat = min(range(len(probabilities)), key=values.__getitem__)

        fh_output.write(f"{id}\t{orgname}\t{site}\t{predict_cat}\t{category}\t{url}\t{probabilities}\n")
    fh_output.close()


def train(args):
    config = load_config(args.config)
    config['action'] = 'train'
    model = Model(config)
    LOGGER.info("starting training process ...")

    model.build_and_train()
    if model.type.startswith('spacy'):
        model.save(config['model_path'])


def predict(args):
    config = load_config(args.config)
    config['action'] = 'predict'
    model = Model(config)
    LOGGER.info("starting predicting process ...")
    model.load()

    if args.test_set:
        test_sets = args.test_set.split(",")
    else:
        test_sets = config['datasets']['test']

    os.makedirs(args.output_dir, exist_ok=True)

    for data_set in test_sets:
        output_file = os.path.join(args.output_dir, data_set + '.tsv')
        LOGGER.info('process test_set [%s] and save result to [%s]',
                    data_set, output_file)

        process_batch(
            model,
            config['datasets']['test'][data_set],
            output_file,
            config
        )


def get_args():
    '''get arguments'''
    parser = ArgumentParser(description=
            'train a model, or inference use a model', prog='PROG')
    subparsers = parser.add_subparsers(help='supported actions')

    parser_train = subparsers.add_parser('train',
            help='train the model')

    parser_train.add_argument('config', help='config file', type=str)

    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser('predict',
            help='predict for a batch of input')

    parser_predict.add_argument('config', help='config file', type=str)
    parser_predict.add_argument('--test_set',
            help='test set defined in the config file to predict', type=str)
    parser_predict.add_argument('--output_dir', help='output directory',
            type=str, default='res')

    parser_predict.set_defaults(func=predict)

    return parser.parse_args()


if __name__ == "__main__":
    set_logging_level(logging.INFO)
    args = get_args()
    args.func(args)
