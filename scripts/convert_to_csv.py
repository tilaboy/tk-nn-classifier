'''strip trxml files'''
from xml_miner.miner import TRXMLMiner
import os
import csv
import logging
from xml.sax.saxutils import escape
from argparse import ArgumentParser

# data summary:
# - uk: 5000/5000 trxml, filename "st.trxml" or "de.trxml", or dir name "staffing_uk"
# - uk_annotated: 9999 trxml with 8193 with csv:
#   posting_id;advertiser_type;spider_source;advertiser_name;type_name
#   1 => direct, 2 => staffing
# - uk: unidentified 4639: 67b27b100c6647d6b5bf419c15d05a7a.un.trxml
# - uk: random 10769:      32d0ea9d7bac42538cb0d06c95e853ec.r.trxml
#   random_annotated.csv:
#   Document.0.correlationid,derived_org_name.0.derived_org_name,new,old,annotation,compare old,compare new,derived_source_site.0.derived_source_site,derived_norm_url.0.derived_norm_url,probabilities
# - us
#   id,advertiser_name,advertiser_type,date,full_text,posting_id,source_type,source_url,source_website,spider_source
#   advertiser_type: csv filename, staffing_us, direct_us
# - all_en
#   num,date,full_text,organization_name,posting_id,source_type,source_url,source_website
#   type: filename


# design notes, how to aggreate all data:
# - folder name: uk,
# - filename: us, all_en
# - mapping to csv: uk_annoated, uk_random
# - how about the rest: unidentified, rest of random


datasets = {
    'uk_staffing_agency': {'country': 'uk', 'clue': 'folder_name'},
    'uk_directed_employer': {'country': 'uk', 'clue': 'folder_name'},
    'unidentified': {'country': 'uk', 'clue': None },
    'annotated': {'country': 'uk', 'clue': 'anno_csv', 'anno_csv': 'annotated_sample_overview.csv'},
    'random': {'country': 'uk', 'clue': 'anno_csv', 'anno_csv': 'random_annotated.csv'},
    'direct_us.csv': {'country': 'us', 'clue': 'file_name'},
    'staffing_us.csv': {'country': 'us', 'clue': 'file_name'},
    'all_en/AT_129_staffing_postings.csv': {'country': 'at', 'clue': 'file_name'},
    'all_en/AT_490_direct_postings.csv': {'country': 'at', 'clue': 'file_name'},
    'all_en/BE_385_staffing_postings.csv': {'country': 'be', 'clue': 'file_name'},
    'all_en/BE_1000_direct_postings.csv': {'country': 'be', 'clue': 'file_name'},
    'all_en/CA_597_staffing_postings.csv': {'country': 'ca', 'clue': 'file_name'},
    'all_en/CA_1000_direct_postings.csv': {'country': 'ca', 'clue': 'file_name'},
    'all_en/DE_618_staffing_postings.csv': {'country': 'de', 'clue': 'file_name'},
    'all_en/DE_1000_direct_postings.csv': {'country': 'de', 'clue': 'file_name'},
    'all_en/FR_225_staffing_postings.csv': {'country': 'fr', 'clue': 'file_name'},
    'all_en/FR_1000_direct_postings.csv': {'country': 'fr', 'clue': 'file_name'},
    'all_en/ES_210_staffing_postings.csv': {'country': 'es', 'clue': 'file_name'},
    'all_en/ES_807_direct_postings.csv': {'country': 'es', 'clue': 'file_name'},
    'all_en/IT_137_staffing_postings.csv': {'country': 'it', 'clue': 'file_name'},
    'all_en/IT_744_direct_postings.csv': {'country': 'it', 'clue': 'file_name'},
    'all_en/NL_765_staffing_postings.csv': {'country': 'nl', 'clue': 'file_name'},
    'all_en/NL_1000_direct_postings.csv': {'country': 'nl', 'clue': 'file_name'}
}

csv_csv_field_mapper = {
    'id': 'id',
    'date': 'date',
    'full_text': 'full_text',
    'advertiser_name': 'organization_name',
    'organization_name': 'organization_name',
    'advertiser_type': 'advertiser_type',
    'posting_id': 'posting_id',
    'source_type': 'source_type',
    'source_url': 'source_url',
    'source_website': 'source_website'
}

csv_trxml_field_mapper = {
    'date': 'derived_vac_posted_date.0.derived_vac_posted_date',
    'full_text': 'sec_vacancy.0.sec_vacancy',
    'organization_name': 'derived_org_name_norm.0.derived_org_name_norm',
    'posting_id': 'Document.0.correlationid',
    'source_type': 'derived_vac_intermediary.0.derived_vac_intermediary',
    'source_url': 'source_url.0.source_url',
    'source_website': 'derived_source_site.0.derived_source_site'
}



def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='''generate csv files from trxml batch:''',
                            prog='PROG')

    parser.add_argument('input_dir', help='input dir contains all data resources', type=str)
    parser.add_argument('--data_sets', help='comma separated file/dir names to be imported', type=str)
    parser.add_argument('--output_file', help='output file', type=str, default='output.csv')

    return parser.parse_args()


def _get_advertiser_type(source):
    # to implement
    return 'yes'

def _check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError('could not find %s', path)

def _load_data(data_path, data_attrib):
    country = data_attrib['country']
    clue = data_attrib['clue']

def main():
    args = get_args()
    trxml_miner = TRXMLMiner(','.join(field_mapper.values()))

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError('could not find %s', args.input_dir)

    for dataset in datasets:
        data_path = os.path.join(args.input_dir, dataset)
        _check_file_path(data_path)
        _load_data(data_path, datasets[dataset])


    rows = []
    index = 0
    for doc in os.listdir(args.input_dir):
        logging.info(doc)
        selected = trxml_miner.mine(os.path.join(args.input_dir, doc))
        values = list(selected)[0]
        row = {'id': index, 'advertiser_type': _get_advertiser_type(args)}

        for csv_field, trxml_field in field_mapper.items():
            row[csv_field] = values['values'][trxml_field]
        rows.append(row)
        index += 1


    with open(args.output_file, 'w', newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['index'] + list(field_mapper.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    main()
