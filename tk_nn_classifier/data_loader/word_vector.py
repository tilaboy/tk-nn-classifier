'''Basic class for word embedding'''

import struct
import mimetypes
import numpy as np


class WordVector:
    '''
    word embedding class:
    it is created from the word2vec type word_embedding, and it contains
    - vocab
    - vectors
    '''
    # Special symbols for padding and unknown word
    PAD = "xxPADxx"
    UNK = "xxUNKxx"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, inputfile):
        '''
        word_vector object
        build from a either a wordvector binary or txt file

        params:
            inputfile: a single filepath as string
        '''
        vocab, vectors = self.read_embeddings(inputfile)
        self.vocab = vocab
        self.vectors = vectors
        self.vocab_to_index = self.create_vocab_index_dict(self.vocab)

    @staticmethod
    def create_vocab_index_dict(vocab):
        vocab_to_index = {}
        for index, word in enumerate(vocab):
            vocab_to_index[word] = index
        return vocab_to_index

    @property
    def vocab_size(self):
        '''
        get the number of tokens in vocabulary
        '''
        return self.vocab.shape[0]

    @property
    def vector_size(self):
        '''
        get the length of the word embedding
        '''
        return self.vectors.shape[1]

    @property
    def unk_vector(self):
        '''
        get the default vector for the unkown word
        '''
        return self.vectors[self.UNK_ID]

    def get_index(self, word):
        '''
        lookup the index given the word in the embedding
        '''
        index = self.UNK_ID
        if word in self.vocab_to_index:
            index = self.vocab_to_index[word]
        return index

    def get_vector(self, word):
        '''
        lookup the vector given the word in the embedding
        '''
        vector_index = self.get_index(word)
        if vector_index < self.vocab_size:
            vector = self.vectors[vector_index]
        return vector

    def get_word(self, index):
        '''
        look up the token in vocabulary with given index
        '''
        return self.vocab[index]

    def __contains__(self, word):
        return word in self.vocab

    def cosine_nearest_neighbors(self, input_vector, nr_neighbors=10):
        '''
        compute the nearest n neighbours of any input vector
        '''
        metrics = np.dot(self.vectors, input_vector.T)
        best = np.argsort(metrics)[::-1][1:nr_neighbors + 1]
        best_metrics = metrics[best]
        return best, best_metrics

    def save_sublist(self, words, output_file):
        """
        Generate a smaller binary word-embeddings model file
        with only the given list of words.
        """

        known_words = [
            w.upper()
            for w in words
            if self.get_index(w.upper()) >= 2
        ]
        nwords = len(known_words)

        with open(output_file, 'wb') as ostream:
            # store header
            ostream.write("{} {}\n".format(nwords,
                                           self.vector_size).encode('ascii'))

            # store word and word_vector
            for word in known_words:
                ostream.write("{} ".format(word).encode('utf-8'))
                ostream.write(
                    struct.pack("f" * self.vector_size,
                                *self.get_vector(word)))
                ostream.write(" ".encode('utf-8'))

    @classmethod
    def read_embeddings(cls, inputfile, vacab_unicode_size=78):
        '''
        Read embeddings files and return a vocabulary and vectors array.

        params:
            inputfile: a single filepath as string
            vacab_unicode_size: max length of words in vocab (chars)

        returns:
            vocab: [vocab_size, vacab_unicode_size] unicode array
            vectors: [vocab_size, vector_size] float array containing
                     embeddings
        '''

        # Read headers to get vocab and vector size
        vocab_size, vector_size = cls.read_embeddings_header(inputfile)
        vocab_size += 2  # +2 for pad and unknown token

        # Create vocab and vector arrays
        vocab = np.empty(vocab_size, dtype='<U%s' % vacab_unicode_size)
        vectors = np.empty((vocab_size, vector_size), dtype=np.float32)

        # Add padding and unknown token
        vocab[0] = cls.PAD
        vocab[1] = cls.UNK
        vectors[0] = np.zeros(vector_size)
        vectors[1] = np.zeros(vector_size)

        mimetype = mimetypes.guess_type(inputfile)
        if mimetype[0] == "text/plain":
            cls._load_embeddings_from_text(inputfile, vocab[2:], vectors[2:])
        else:
            cls._load_embeddings_from_binary(inputfile, vocab[2:], vectors[2:])

        return vocab, vectors

    @classmethod
    def read_embeddings_header(cls, inputfile):
        '''
        read the header from the file, note that both binary and text have
        the same header.
        '''
        mimetype = mimetypes.guess_type(inputfile)
        if mimetype[0] == "text/plain":
            readmode = 'r'
            encoding = 'utf-8'
        else:
            readmode = 'rb'
            encoding = None
        with open(inputfile, readmode, encoding=encoding) as fin:
            header = fin.readline()
            vocab_size, vector_size = list(map(int, header.split()))
        return vocab_size, vector_size

    @classmethod
    def _load_embeddings_from_binary(cls, filename, vocab, vectors):

        vocab_size, vector_size = cls.read_embeddings_header(filename)

        assert vocab.shape[0] == vocab_size
        assert vectors.shape[0] == vocab_size
        assert vectors.shape[1] == vector_size

        with open(filename, 'rb') as fin:
            _ = fin.readline()  # first line is header
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for i in range(vocab_size):
                word = b''
                cur_char = fin.read(1)
                while cur_char != b' ':
                    word += cur_char
                    cur_char = fin.read(1)
                vector = np.frombuffer(fin.read(binary_len), dtype=np.float32)
                fin.read(1)
                vocab[i] = word.decode('utf-8')
                vectors[i] = unitvec(vector)
        print(
            "read {} tokens with vector size {} from {}".format(
                vocab_size,
                vector_size,
                filename))

    @classmethod
    def _load_embeddings_from_text(cls, filename, vocab, vectors):

        vocab_size, vector_size = cls.read_embeddings_header(filename)

        assert vocab.shape[0] == vocab_size
        assert vectors.shape[0] == vocab_size
        assert vectors.shape[1] == vector_size

        with open(filename, 'r', encoding='utf-8') as fin:
            _ = fin.readline()
            for index, line in enumerate(fin):
                line = line.strip()
                parts = line.split(' ')
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float)
                vocab[index] = word
                vectors[index] = unitvec(vector)
        print(
            "read {} tokens with vector size {} from {}".format(
                vocab_size,
                vector_size,
                filename))


def unitvec(vec):
    '''
    normalize the vector
    '''
    return (1.0 / np.linalg.norm(vec, ord=2)) * vec


def maxabs(embeddings, axis=0):
    """Return slice of embeddings, keeping only those values that are
    furthest away from 0 along axis"""
    maxa = embeddings.max(axis=axis)
    mina = embeddings.min(axis=axis)
    positive = abs(maxa) >= abs(mina)  # bool, or indices where +ve values win
    negative = abs(mina) >= abs(maxa)  # bool, or indices where -ve values win
    if axis is None:
        if positive:
            return maxa
        else:
            return mina
    shape = list(embeddings.shape)
    shape.pop(axis)
    out = np.zeros(shape, dtype=embeddings.dtype)
    out[positive] = maxa[positive]
    out[negative] = mina[negative]
    return out
