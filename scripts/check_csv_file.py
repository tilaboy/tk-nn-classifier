import csv
from argparse import ArgumentParser

def _prepare_input_text(text):
    lines = text.split("\n")
    text = "\n".join(lines[:500])
    return text

def check_staffing_agency_csv(data_path):
    total_counts = dict()
    with open(data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = _prepare_input_text(row['full_text'])
            adv_type = row['advertiser_type']
            total_counts[adv_type] = total_counts.get(adv_type, 0) + 1
            if len(text) < 10 or row['advertiser_type'] not in ['1', '2']:
                print(row)
    return total_counts

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
    total_counts = check_staffing_agency_csv(args.csv_file)

    for category in total_counts:
        print("type {}: in total {} items".format(category, total_counts[category]))
