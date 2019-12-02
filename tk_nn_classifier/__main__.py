'''train and process a batch of documents'''
from __future__ import unicode_literals, print_function
import os
from argparse import ArgumentParser
import logging
import csv
from tk_nn_classifier.model import Model
from tk_nn_classifier.config import load_config
from tk_nn_classifier.data_loader import DataReader
from tk_nn_classifier import set_logging_level, LOGGER


def process_batch(model, reader, data_set, config):
    result = []
    input_data = reader.get_data_set_with_detail(
            config['datasets']['test'][data_set]
    )

    detail_fields = reader._detail_fields(config['datasets']['test'][data_set])
    header = [detail_fields[2], 'new',  'old'] + detail_fields[3:] + \
             ['probabilities']
    result.append(header)
    for test_text, category, id, *extra in input_data:
        probabilities = model.process_with_saved_model(test_text)
        if type(probabilities) is list:
            predicted_class = max(range(len(probabilities)),
                                  key=probabilities.__getitem__)
            predicted_class = reader.label_mapper.label_name(predicted_class)
        elif type(probabilities) is dict:
            predicted_class = max(probabilities, key=probabilities.get)
        else:
            raise ValueError("unknown type", type(probabilities))

        result.append(
            [
                entry if entry is not None else ''
                for entry in [id, predicted_class,
                              category, *extra, str(probabilities)]
            ]
        )
    return result


def train(args):
    config = load_config(args.config)
    config['action'] = 'train'
    model = Model(config)
    LOGGER.info("starting training process ...")

    model.build_and_train()


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

    data_reader = DataReader(model.config)
    os.makedirs(args.output_dir, exist_ok=True)

    for data_set in test_sets:
        output_file = os.path.join(args.output_dir, data_set + '.tsv')
        LOGGER.info('process test_set [%s]', data_set)
        result = process_batch(
            model,
            data_reader,
            data_set,
            config
        )
        LOGGER.info('save result to [%s]', output_file)
        with open(output_file, 'w', newline='') as output_fh:
            csv_writer = csv.writer(output_fh,
                                    delimiter="\t",
                                    quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerows(result)


def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='train a classifier, or predict class',
                            prog='PROG')
    subparsers = parser.add_subparsers(help='supported actions')

    parser_train = subparsers.add_parser('train',
                                         help='train the model')

    parser_train.add_argument('config', help='config file', type=str)

    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser('predict',
                                           help='predict for a batch of input')

    parser_predict.add_argument('config', help='config file', type=str)
    parser_predict.add_argument('--test_set',
                                help='name of test set in the config file',
                                type=str)
    parser_predict.add_argument('--output_dir',
                                help='output directory',
                                type=str, default='res')

    parser_predict.set_defaults(func=predict)

    return parser.parse_args()


def main():
    set_logging_level(logging.INFO)
    args = get_args()
    args.func(args)


if __name__ == "__main__":
    main()
