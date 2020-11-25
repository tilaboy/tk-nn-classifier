from argparse import ArgumentParser
import csv

from tk_nn_classifier.data_loader import WordVector, tokenize

def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='filter word embedding on given tokens')
    parser.add_argument('embedding_file', help='original embedding file path')
    parser.add_argument('csv_files', help='csv files with text, seperated with ","')
    parser.add_argument('sub_embedding_file', help= 'embedding only with words from list')
    parser.add_argument('--csv_field', help= 'csv file name with target text', default='full_text')
    return parser.parse_args()

def main():
    '''remove all tokens from word embedding except ones in the text of given csv files.'''
    args = get_args()
    embedding = WordVector(args.embedding_file)
    tokens = tokens_from_csvfiles(args.csv_files, args.csv_field)
    embedding.save_sublist(tokens, args.sub_embedding_file)


def tokens_from_csvfiles(files, field_name):
    '''get token list from csv files'''
    token_dict = {}
    for file in files.split(','):
        _tokens_from_csvfile(token_dict, file, field_name)
    return sorted(list(token_dict.keys()))


def _tokens_from_csvfile(token_dict, input_csv_file:str, field_name:str):
    with open(input_csv_file, newline='', encoding='utf-8-sig') as csv_fh:
        reader = csv.DictReader(csv_fh)
        for row in reader:
            for token in tokenize(row[field_name]):
                token_dict[token] = token_dict.get(token, 0) + 1


if __name__ == '__main__':
    main()
