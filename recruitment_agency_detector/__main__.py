from __future__ import unicode_literals, print_function
import os
from argparse import ArgumentParser
import logging
from recruitment_agency_detector.model import Model
from recruitment_agency_detector.config import load_config
from recruitment_agency_detector.data_loader import DataReader
from recruitment_agency_detector import set_logging_level, LOGGER

def process_batch(model, reader, data_set, config):
    result = []
    input_data = reader.get_data_set_with_detail(
            config['datasets']['test'][data_set]
    )

    detail_fields = reader._detail_fields(config['datasets']['test'][data_set])
    header = [detail_fields[2], detail_fields[1] + '_new',  detail_fields[1] ] + detail_fields[3:] + ['probablities']
    result.append(header)
    for test_text, category, id, *extra in input_data:
        probabilities = model.process_with_saved_model(test_text)
        # todo: this is the index of classes, still need to map back
        predicted_class = max(probabilities, key=probabilities.get)
        #predicted_label = class_to_label()
        result.append(
            [id, predicted_class, category, *extra, str(probabilities)]
        )
    return result

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

    data_reader = DataReader(config)

    for data_set in test_sets:
        LOGGER.info('process test_set [%s]', data_set)
        result = process_batch(
            model,
            data_reader,
            data_set,
            config
        )
        write_to_output(
            result,
            args.output_dir,
            data_set
        )

def write_to_output(result, output_dir, data_set):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, data_set + '.tsv')
    with open(output_file, 'w') as fh_output:
        for line in result:
            fh_output.write("\t".join(line) + "\n")


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


def main():
    set_logging_level(logging.INFO)
    args = get_args()
    args.func(args)


if __name__ == "__main__":
    main()
