from xml_miner.miner import TRXMLMiner

def get_data_from_trxml(data_dir):
    fields = [
        'sec_vacancy.0.sec_vacancy',
        'derived_source_type.0.derived_source_type'
    ]

    trxml_miner = TRXMLMiner(','.join(fields))
    data = []
    for trxml in trxml_miner.mine(data_dir):
        sec_vacancy = trxml['values']['sec_vacancy.0.sec_vacancy']
        type = trxml['values']['derived_source_type.0.derived_source_type']
        if type == 'wervenuitzendsite':
            category = 1
        elif type == 'other':
            category = 0
        else:
            category = 0.5
        data.append( (_prepare_input_text(sec_vacancy), category) )
    return data


def _prepare_input_text(text):
    lines = text.split("\n")
    return "\n".join(lines[:50])


def get_data_with_details(data_dir):
    fields = [
        'sec_vacancy.0.sec_vacancy',
        'derived_source_type.0.derived_source_type',
        'Document.0.correlationid',
        'derived_org_name.0.derived_org_name',
        'derived_source_site.0.derived_source_site',
        'derived_norm_url.0.derived_norm_url'
    ]
    trxml_miner = TRXMLMiner(','.join(fields))

    for trxml in trxml_miner.mine(data_dir):
        trxml['values']['sec_vacancy.0.sec_vacancy'] = _prepare_input_text(
                trxml['values']['sec_vacancy.0.sec_vacancy'])
        yield (trxml['values'][field] for field in fields)
