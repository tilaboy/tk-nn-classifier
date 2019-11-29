'''strip trxml files'''
from xml_miner.miner import TRXMLMiner
import os
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from argparse import ArgumentParser


def _trxml_template(values):
    trxml_string = '''<?xml version="1.0" encoding="UTF-8" ?>
<TextractorResult content_type="text/xml" source="text/xml">
<DocumentStructure>
'''

    for field in values:
        (itemgroup, index, field_name) = field.split('.')
        text = escape(values[field])
        trxml_string += f'''    <ItemGroup key="{itemgroup}">
        <Item index="{index}">
            <Field key="{field_name}"><Value>{text}</Value></Field>
        </Item>
    </ItemGroup>
'''

    trxml_string_post='''
</DocumentStructure>
</TextractorResult>
'''

    trxml_string += trxml_string_post
    return trxml_string


def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='''strip trxml files:
             clean up the unused field to have a smaller trxml,
             after the script, trxml will only have necessary fields
             and remove all others

             Note: only work with singleton fields''', prog='PROG')

    parser.add_argument('input_dir', help='input dir contains trxml to clean', type=str)
    parser.add_argument('--output_dir', help='output directory', type=str, default=None)

    return parser.parse_args()


def _fields_to_keep():
    fields = ["sec_vacancy.0.sec_vacancy",
              "derived_vac_intermediary.0.derived_vac_intermediary",
              "Document.0.correlationid",
              "derived_org_name.0.derived_org_name",
              "derived_source_site.0.derived_source_site",
              "derived_norm_url.0.derived_norm_urltest",
              "derived_cond_contract_type.0.derived_cond_contract_type"
             ]
    return fields


def main():
    args = get_args()
    trxml_miner = TRXMLMiner(','.join(_fields_to_keep()))

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    for selected_values in trxml_miner.mine(args.input_dir):
        doc_id =selected_values['values']['Document.0.correlationid']
        trxml_string = _trxml_template(selected_values['values'])
        try:
            tree = ET.ElementTree(ET.fromstring(trxml_string))
        except Exception as e:
            print(doc_id, e)

        output_file = os.path.join(output_dir, doc_id + '.un.trxml')
        with open(output_file, 'w', encoding='utf8') as fh:
            fh.write(trxml_string)


if __name__ == '__main__':
    main()
