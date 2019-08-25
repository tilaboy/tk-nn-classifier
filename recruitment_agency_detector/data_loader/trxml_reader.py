import random
from xml_miner.miner import TRXMLMiner


def _get_values_from_trxml(fields, data_dir):
    # the first element in the fields is the input text to the data models
    trxml_miner = TRXMLMiner(','.join(fields))
    data = []
    for trxml in trxml_miner.mine(data_dir):
        trxml['values'][fields[0]] = _prepare_input_text(
                trxml['values'][fields[0]])
        yield [trxml['values'][field] for field in fields]


def _prepare_input_text(text):
    lines = text.split("\n")
    return "\n".join(lines[:50])


def _get_data_from_trxml(data_dir):
    fields = [
        'sec_vacancy.0.sec_vacancy',
        'derived_vac_intermediary.0.derived_vac_intermediary'
    ]
    return _get_values_from_trxml(fields, data_dir)

def get_spacy_data(data_dir, shuffle=False, train_mode=False):
    data_set = list(_get_data_from_trxml(data_dir))
    if shuffle:
        random.shuffle(data_set)
    texts, labels = zip(*data_set)
    cats = [{"yes": label == "yes", "no": label == "no"} for label in labels]
    if train_mode:
        cats = [{"cats": cat} for cat in cats]

    return (list(zip(texts, cats)))




def get_data_with_details(data_dir):
    fields = [
        'sec_vacancy.0.sec_vacancy',
        'derived_vac_intermediary.0.derived_vac_intermediary',
        'Document.0.correlationid',
        'derived_org_name.0.derived_org_name',
        'derived_source_site.0.derived_source_site',
        'derived_norm_url.0.derived_norm_url'
    ]
    return _get_values_from_trxml(fields, data_dir)
