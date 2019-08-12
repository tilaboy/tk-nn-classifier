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
import plac
import random
from pathlib import Path

import spacy

from xml_miner.miner import TRXMLMiner
from graph import Graph
from train import Trainer

def main(model="en_core_web_sm", output_dir='output_dir', n_iter=10, init_tok2vec=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    # add graph to spacy
    nlp = Graph.build_spacy_graph(model)

    # load the IMDB dataset
    print("Loading data...")
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data()
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    nlp = Trainer.train_spacy(nlp, train_data, dev_texts, dev_cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

def inference(output_dir='output_dir'):
    # test the trained model

    print("Loading from", output_dir)
    nlp = spacy.load(output_dir)

    fh_train_output = open('train.txt', 'w')
    for test_text, category in get_data('data/trxml/'):
        doc = nlp(test_text)
        fh_train_output.write(str(category) + " <=> " + str(doc.cats) + "\n")
    fh_train_output.close()

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


def evaluate_and_print(nlp=None, output_dir='output_dir'):
    if nlp is None:
        print("Loading from", output_dir)
        nlp = spacy.load(output_dir)

    train_data_path = 'data/trxml'
    train_data = get_data(train_data_path)
    train_texts, train_labels = zip(*train_data)
    print('compute scores on training set')
    compute_score(nlp, train_texts, train_labels)

    eval_data_path = 'data/unidentified'

    eval_data = get_data(eval_data_path)
    eval_texts, _ = zip(*eval_data)
    eval_labels = [1] * len(eval_texts)
    print('compute scores on eval set')
    compute_score(nlp, eval_texts, eval_labels)


def get_data(data_dir):
    trxml_miner = TRXMLMiner("fulltext.0.fulltext,derived_source_type.0.derived_source_type")
    data = []

    for mined in trxml_miner.mine(data_dir):
        if mined['values']['derived_source_type.0.derived_source_type'] == 'wervenuitzendsite':
            category = 1
        elif mined['values']['derived_source_type.0.derived_source_type'] == 'other':
            category = 0
        else:
            category = 0.5
        data.append( (mined['values']['fulltext.0.fulltext'], category) )
    return data

def load_data(split=0.8, limit=0):
    """Load data from our dataset."""
    # Partition off part of the train data for evaluation
    # train_data, _ = thinc.extra.datasets.imdb()
    # vacs_df = pandas.read_json(file_path)

    train_data_path = 'data/trxml_small'

    train_data = get_data(train_data_path)
    random.shuffle(train_data)
    train_data = train_data[-limit:]


    texts, labels = zip(*train_data)

    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]

    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


if __name__ == "__main__":
    #plac.call(evaluate_and_print)
    plac.call(main)
