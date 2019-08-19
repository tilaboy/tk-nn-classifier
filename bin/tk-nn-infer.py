from __future__ import unicode_literals, print_function

import spacy
from argparse import ArgumentParser
from recruitment_agency_detector.utils import get_data_with_details


def main(model_dir, test_dir, output_file='result.txt'):
    print("Loading from", model_dir)
    nlp = spacy.load(model_dir)

    fh_output = open(output_file, 'w')
    fh_output.write('id\torg_name\tsite\tnew_predict\told_predict\turl\tscore\n')
    for test_text, category, id, orgname, site, url in get_data_with_details(test_dir):
        doc = nlp(test_text)
        predict_cat = 'yes' if doc.cats['yes'] > doc.cats['no'] else 'no'
        fh_output.write(f"{id}\t{orgname}\t{site}\t{predict_cat}\t{category}\t{url}\t{doc.cats}\n")
    fh_output.close()

def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='''
                            the skill validation model
                            ''')
    parser.add_argument('model_dir', help='training config file', type=str)
    parser.add_argument('test_dir', help='directory with test files', type=str)
    parser.add_argument('output_file', help='file to save the output results', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.model_dir, args.test_dir, args.output_file)
