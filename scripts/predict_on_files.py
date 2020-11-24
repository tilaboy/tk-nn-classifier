import os
from tensorflow.contrib import predictor
from xml_miner.miner import TRXMLMiner
from argparse import ArgumentParser
from tk_nn_classifier.data_loader import WordVector
from easy_tokenizer.tokenizer import Tokenizer
from tensorflow.python.keras.preprocessing import sequence

MAXLEN = 512
TEXT_FIELD = 'sec_vacancy.0.sec_vacancy'

def _input_text_to_pad_id(text, vocab_to_ids, tokenizer):
    data_id = [vocab_to_ids[token]
               if token in vocab_to_ids else WordVector.UNK_ID
               for token in tokenizer.tokenize(text.upper())
              ]
    data = sequence.pad_sequences([data_id],
                                  maxlen=MAXLEN,
                                  truncating='post',
                                  padding='post',
                                  value=WordVector.PAD_ID)
    return {'input':data}


def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='process trxml file/files:')
    parser.add_argument('input', help='input trxml/trxml folder to predict', type=str)
    parser.add_argument('model_path', help='trained classifier', type=str)
    parser.add_argument('embedding_path', help='path of the embedding file', type=str)
    return parser.parse_args()


def _load_model_and_vocab(args):
    model = predictor.from_saved_model(args.model_path)
    vocab, _ = WordVector.read_embeddings(args.embedding_path)
    vocab_to_ids = WordVector.create_vocab_index_dict(vocab)
    return model, vocab_to_ids


def main():
    args = get_args()
    trxml_miner = TRXMLMiner(TEXT_FIELD)

    if os.path.isdir(args.input):
        files = [os.path.join(args.input, file) for file in os.listdir(args.input)]
    elif os.path.isfile(args.input):
        files = [args.input]
    else:
        raise ValueError('Input not exist')

    model, vocab_to_ids = _load_model_and_vocab(args)
    tokenizer = Tokenizer()

    for file in files:
        selected_value = list(trxml_miner.mine(file))
        input_text = selected_value[0]['values'][TEXT_FIELD]
        data = _input_text_to_pad_id(input_text, vocab_to_ids, tokenizer)
        result = model(data)
        probabilities = result['probabilities'][0]
        print(file, probabilities)


if __name__ == '__main__':
    main()
