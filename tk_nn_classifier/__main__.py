'''train and process a batch of documents'''
from __future__ import unicode_literals, print_function
import os
from argparse import ArgumentParser
import logging
from .model import Model
from .config import load_config
from .data_loader import load_data_set, split_data_set
from . import set_logging_level, LOGGER


def train(args):
    # load config
    config = load_config(args.config)
    config['action'] = 'train'

    # split data if needed
    if 'all_data' in config['datasets']:
        config['datasets']['train'], config['datasets']['eval'] = \
            split_data_set(config['datasets']['all_data'])

    # load_data
    train_raw = load_data_set(config, config['datasets']['train'])
    eval_raw = load_data_set(config, config['datasets']['eval'])

    # declare model
    model = Model(config)

    # transfer data set [{f1: v1, f2: v2, ...}, {}, ...]
    # to model input, e.g. [[feature], [label]]
    train_data = model.prepare_input(train_raw, train_mode=True)
    eval_data = model.prepare_input(eval_raw, train_mode=False)

    # build graph
    model.build_graph()

    # train
    model.train(train_data, eval_data)

    # save the model
    model.save(config['model_path'])

    # test if test in datasets
    for testset in config['datasets'].get('test', []):
        test_file = config['datasets']['test'][testset]
        LOGGER.info('eval on test {} : {}'.format(testset, test_file))
        model.eval_test_set(test_file)

def eval(args):
    config = load_config(args.config)
    config['action'] = 'predict'
    model = Model(config)
    model.load()

    if args.test_set:
        test_sets = args.test_set.split(",")
    else:
        test_sets = config['datasets']['test']

    os.makedirs(config['model_eval_path'], exist_ok=True)

    for testset in test_sets:
        test_file = config['datasets']['test'][testset]
        output_file = os.path.join(config['model_eval_path'], testset + '.tsv')
        LOGGER.info('eval dataset {} ({}), and save to {}'.format(testset,
                                                           test_file,
                                                           output_file))
        model.eval_test_set(test_file, analysis_output_file=output_file)

def predict(args):
    # TODO: convert the script predict to here
    pass


def _get_column(matrix, column_i):
    return [matrix[i][column_i] for i in range(1, len(matrix))]

def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='train a classifier, or predict class')

    subparsers = parser.add_subparsers(help='supported actions')
    parser_train = subparsers.add_parser('train', help='train the model')
    parser_train.add_argument('config', help='config file', type=str)
    parser_train.set_defaults(func=train)

    parser_eval = subparsers.add_parser('eval', help='eval on all test sets')
    parser_eval.add_argument('config', help='config file', type=str)
    parser_eval.add_argument('--test_set',
                             help='name of test set in the config file',
                             type=str)
    parser_eval.add_argument('--output_dir_name',
                             help='output directory name, saved as model_path/output_dir_name',
                             type=str, default='res')
    parser_eval.set_defaults(func=eval)

    parser_predict = subparsers.add_parser('predict',
                                           help='predict for a batch of input')
    parser_predict.add_argument('config', help='config file', type=str)
    parser_predict.add_argument('--input_file_path',
                                help='name of test set in the config file',
                                type=str)
    parser_predict.add_argument('--output_file_path',
                                help='result output file',
                                type=str, default='output.csv')
    parser_predict.set_defaults(func=predict)

    return parser.parse_args()


def main():
    set_logging_level(logging.INFO)
    args = get_args()
    args.func(args)


if __name__ == "__main__":
    main()
