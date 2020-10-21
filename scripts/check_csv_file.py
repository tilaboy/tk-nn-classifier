import csv
from argparse import ArgumentParser

def _prepare_input_text(text):
    lines = text.split("\n")
    text = "\n".join(lines[:500])
    return text

def check_staffing_agency_csv(data_path):
    with open(data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = _prepare_input_text(row['full_text'])
            if len(text) < 10 or row['advertiser_type'] not in ['1', '2']:
                print(row)

def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='check staffing agency input csv',
                            prog='PROG')
    parser.add_argument('csv_file',
                        help='csv file needs to be checked',
                        type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    check_staffing_agency_csv(args.csv_file)
