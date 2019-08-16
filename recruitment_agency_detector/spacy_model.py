#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
import spacy
import random
from xml_miner.miner import TRXMLMiner

def predict_trxml_batch(model_dir='first_model', output_file='result.txt'):
    print("Loading from", model_dir)
    nlp = spacy.load(model_dir)

    fh_output = open(output_file, 'w')
    fh_output.write('id\torg_name\tsite\tnew_predict\told_predict\turl\tscore\n')
    for test_text, category, id, orgname, site, url in get_data_with_details('data/random_trxmls'):
        doc = nlp(test_text)
        predict_cat = 1 if doc.cats['POSITIVE'] > doc.cats['NEGATIVE'] else 0
        fh_output.write(f"{id}\t{orgname}\t{site}\t{predict_cat}\t{category}\t{url}\t{doc.cats}\n")
    fh_output.close()

def evaluate_trxml_batch(model_path, data_path):
    nlp = spacy.load(model_path)
    train_data = get_data(data_path)
    train_texts, train_labels = zip(*train_data)
    print('compute scores on training set')
    compute_score(nlp, train_texts, train_labels)


def compute_score(nlp, texts, labels):

    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]

    textcat = nlp.get_pipe("textcat")
    optimizer = nlp.begin_training()

    with textcat.model.use_params(optimizer.averages):
        # evaluate on the dev data split off in load_data()
        scores = evaluate(nlp.tokenizer, textcat, texts, cats)
    print(
        "{0:.3f}\t{1:.3f}\t{2:.3f}".format(  # print a simple table
            scores["textcat_p"],
            scores["textcat_r"],
            scores["textcat_f"],
        )
    )


def get_data_with_details(data_dir):
    fields = [
        'fulltext.0.fulltext',
        'derived_source_type.0.derived_source_type',
        'Document.0.correlationid',
        'derived_org_name.0.derived_org_name',
        'derived_source_site.0.derived_source_site',
        'derived_norm_url.0.derived_norm_url'
    ]
    trxml_miner = TRXMLMiner(','.join(fields))
    data = []

    for mined in trxml_miner.mine(data_dir):
        if mined['values']['derived_source_type.0.derived_source_type'] == 'wervenuitzendsite':
            category = 1
        elif mined['values']['derived_source_type.0.derived_source_type'] == 'other':
            category = 0
        else:
            category = 0.5
        data.append(
                    (mined['values']['sec_vacancy.0.sec_vacancy'],
                     category,
                     mined['values']['Document.0.correlationid'],
                     mined['values']['derived_org_name.0.derived_org_name'],
                     mined['values']['derived_source_site.0.derived_source_site'],
                     mined['values']['derived_norm_url.0.derived_norm_url']
                     )
                    )
    return data


def get_data(data_dir):
    trxml_miner = TRXMLMiner("sec_vacancy.0.sec_vacancy,derived_source_type.0.derived_source_type")
    data = []
    for mined in trxml_miner.mine(data_dir):
        if mined['values']['derived_source_type.0.derived_source_type'] == 'wervenuitzendsite':
            category = 1
        elif mined['values']['derived_source_type.0.derived_source_type'] == 'other':
            category = 0
        else:
            category = 0.5
        data.append( (mined['values']['sec_vacancy.0.sec_vacancy'], category) )
    return data


def load_data(config):
    """Load data from our dataset."""
    train_data = get_data(config['train_data_path'])
    random.shuffle(train_data)
    texts, labels = zip(*train_data)
    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
    split = int(len(train_data) * config['split_ratio'])
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])
