'''strip trxml files'''
from xml_miner.miner import TRXMLMiner
import os
import sys
import csv
import logging
from xml.sax.saxutils import escape
from argparse import ArgumentParser
import hashlib
import re

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
# - folder name: uk (trxml)
# - mapping to csv: uk_annoated, uk_random (trxml)
# - how about the rest: unidentified, rest of random (trxml)
# - filename: us, all_en (csv)

def define_logger(mod_name):
    """Set the default logging configuration"""
    logger = logging.getLogger(mod_name)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(levelname).1s [%(asctime)s] [%(name)s] %(message)s'))
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def set_logging_level(level=logging.WARN):
    """Change logging level"""
    LOGGER.setLevel(level)


LOGGER = define_logger('data_aggre')
fh = logging.FileHandler('data_aggre.log', 'w')
LOGGER.addHandler(fh)


datasets = {
    'uk_staffing_agency': {'country': 'uk', 'clue': 'folder_name'},
    'uk_directed_employer': {'country': 'uk', 'clue': 'folder_name'},
    #'unidentified': {'country': 'uk', 'clue': None },
    'annotated': {'country': 'uk', 'clue': 'anno_csv', 'anno_csv': 'annotated_sample_overview.csv'},
    'random': {'country': 'uk', 'clue': 'anno_csv', 'anno_csv': 'random_annotated.csv'},
    'direct_us.csv': {'country': 'us', 'clue': 'file_name'},
    'staffing_us.csv': {'country': 'us', 'clue': 'file_name'},
    'AT_129_staffing_postings.csv': {'country': 'at', 'clue': 'file_name'},
    'AT_490_direct_postings.csv': {'country': 'at', 'clue': 'file_name'},
    'BE_385_staffing_postings.csv': {'country': 'be', 'clue': 'file_name'},
    'BE_1000_direct_postings.csv': {'country': 'be', 'clue': 'file_name'},
    'CA_597_staffing_postings.csv': {'country': 'ca', 'clue': 'file_name'},
    'CA_1000_direct_postings.csv': {'country': 'ca', 'clue': 'file_name'},
    'DE_618_staffing_postings.csv': {'country': 'de', 'clue': 'file_name'},
    'DE_1000_direct_postings.csv': {'country': 'de', 'clue': 'file_name'},
    'FR_225_staffing_postings.csv': {'country': 'fr', 'clue': 'file_name'},
    'FR_1000_direct_postings.csv': {'country': 'fr', 'clue': 'file_name'},
    'ES_210_staffing_postings.csv': {'country': 'es', 'clue': 'file_name'},
    'ES_807_direct_postings.csv': {'country': 'es', 'clue': 'file_name'},
    'IT_137_staffing_postings.csv': {'country': 'it', 'clue': 'file_name'},
    'IT_744_direct_postings.csv': {'country': 'it', 'clue': 'file_name'},
    'NL_765_staffing_postings.csv': {'country': 'nl', 'clue': 'file_name'},
    'NL_1000_direct_postings.csv': {'country': 'nl', 'clue': 'file_name'}
}

output_fields = ['id', 'posting_id', 'country', 'advertiser_type', 'organization_name', 'source_url', 'full_text']

csv_csv_field_mapper = {
    'id': 'id',
    'advertiser_type': 'advertiser_type',
    'date': 'date',
    'full_text': 'full_text',
    #'advertiser_name': 'organization_name',
    'organization_name': 'organization_name',
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


def _check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError('could not find %s', path)

def _load_data(data_path, data_attrib):
    country = data_attrib['country']
    clue = data_attrib['clue']

def _load_trxml(data_path, trxml_miner, data_attrib):
    docs = []
    index = 0
    common_label = None

    if data_attrib['clue'] == 'anno_csv':
        csv_annotation_file = os.path.join(data_path, '..', data_attrib['anno_csv'])
        LOGGER.info('loadding annotated csv file {}'.format(csv_annotation_file))
        annotated_samples = {
            row['posting_id']: 'yes' if row['advertiser_type'] == 'staffing' else 'no'
            for row in _load_csv(csv_annotation_file)
        }
    else:
        common_label = _get_label_from_name(data_path)

    selected = list(trxml_miner.mine(data_path))

    for doc in selected:
        #logging.debug('processing %s' % doc)
        #selected = trxml_miner.mine(os.path.join(data_path, doc))

        if data_attrib['clue'] == 'anno_csv':
            posting_id = doc['values']['Document.0.correlationid']
            if posting_id not in annotated_samples:
                continue
            else:
                advertiser_type = annotated_samples[posting_id]
        else:
            advertiser_type = common_label

        feature = {
            'id': index,
            'advertiser_type': advertiser_type,
            'country': data_attrib['country']
        }

        for csv_field, trxml_field in csv_trxml_field_mapper.items():
            feature[csv_field] = doc['values'][trxml_field]
        docs.append(feature)
        index += 1
    return docs


def _get_label_from_name(name):
    label = None
    if 'staffing' in name:
        label = 'yes'
    elif 'direct' in name:
        label = 'no'
    else:
        raise ValueError('unknown type from name {}'.format(name))
    return label


def _load_csv(data_path):
    with open(data_path, 'r', newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        docs = list(reader)
    return docs

def _load_csv_with_label(data_path, data_attrib):
    docs = _load_csv(data_path)
    label = _get_label_from_name(data_path)
    for doc in docs:
        doc['advertiser_type'] = label
        doc['country'] = data_attrib['country']
    return docs

def _to_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _summarize_on_org_name(loaded_docs):
    # summarize using org_name
    summary_loaded = {}
    for loaded_doc in loaded_docs:
        org_name = loaded_doc['organization_name']
        country = loaded_doc['country']
        if org_name in summary_loaded:
            if country in summary_loaded[org_name]:
                summary_loaded[org_name][country] += 1
            else:
                summary_loaded[org_name][country] = 1
        else:
            summary_loaded[org_name] = {country: 1}
    return summary_loaded

def _write_csv(filename, header, rows):
    with open(filename, 'w', newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = get_args()
    trxml_miner = TRXMLMiner(','.join(csv_trxml_field_mapper.values()))

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError('could not find %s', args.input_dir)

    data = []
    md5_list = {}
    org_name_summary = {}
    new_data_folder = 'loaded_data'
    short_docs = {}

    for dataset in datasets:
        LOGGER.info('processing dataset: {}'.format(dataset))
        data_path = os.path.join(args.input_dir, dataset)
        data_attrib = datasets[dataset]
        _check_path(data_path)
        if os.path.isdir(data_path):
            loaded_docs = _load_trxml(data_path, trxml_miner, data_attrib)
        else:
            loaded_docs = _load_csv_with_label(data_path, data_attrib)

        filtered_loaded_docs = []
        # deduplicated against readed using md5
        for loaded_doc in loaded_docs:
            doc_md5 = _to_md5(loaded_doc['full_text'])
            if doc_md5 in md5_list:
                LOGGER.info('skip duplicated file {} <=> {}'.format(loaded_doc['posting_id'], md5_list[doc_md5]))
                continue
            else:
                if len(loaded_doc['full_text']) < 400:
                    text = loaded_doc['full_text']
                    cleaned_text = re.sub('[^A-Za-z0-9]+', '', text)
                    text = re.sub(r'\n', ' ', text)
                    if cleaned_text not in short_docs:
                        short_docs[clean_text] = {
                            'dataset': dataset,
                            'posint_id': loaded_doc['posting_id'],
                            'full_text': text
                        }
                    #LOGGER.info('small doc {}: {}'.format(loaded_doc['posting_id'], loaded_doc['full_text']))
                    continue

                md5_list[doc_md5] = loaded_doc['posting_id']
                filtered_loaded_docs.append(loaded_doc)

        counts_org_name = _summarize_on_org_name(filtered_loaded_docs)
        for org_name in counts_org_name:
            if org_name in org_name_summary:
                #LOGGER.info('org {} already exist in {}'.format(org_name, org_name_summary[org_name]))
                for country in counts_org_name[org_name]:
                    org_name_summary[org_name][country] = counts_org_name[org_name][country]
            else:
                org_name_summary[org_name] = counts_org_name[org_name]

        os.makedirs(new_data_folder, exist_ok=True)
        _write_csv(os.path.join(new_data_folder, dataset + '.csv'), output_fields, filtered_loaded_docs)
        data.extend(filtered_loaded_docs)



        # splitted data into train/eval
        # - on org_name
        # - annoated_random need to be leaved as test sets

    _write_csv(os.path.join(new_data_folder, 'org_summary.csv'),
               ['org_name', 'country', 'count'],
               [
                   {'org_name': org_name, 'country': country, 'count': org_name_summary[org_name][country]}
                   for org_name in org_name_summary
                   for country in org_name_summary[org_name]
               ]
              )

    org_total_counts = {
        org_name: sum(org_name_summary[org_name].values())
        for org_name in org_name_summary
    }
    org_total_sorted = [
        {'org_name': org_name, 'count': org_total_counts[org_name]}
        for org_name in sorted(org_total_counts, key=org_total_counts.get, reverse=True)
    ]
    _write_csv(os.path.join(new_data_folder, 'org_total_summary.csv'),
               ['org_name', 'count'],
               org_total_sorted
              )

    _write_csv(os.path.join(new_data_folder, 'short_text.csv'),
               ['dataset', 'id', 'cleaned', 'text'],
               [
                   {
                        dataset:short_docs[short_text]['dataset'],
                        id: short_docs[short_text]['posting_id'],
                        cleaned: short_text,
                        text: short_docs[short_text]['full_text']
                   }
                   for short_text in short_docs
               ]
              )


if __name__ == '__main__':
    main()
