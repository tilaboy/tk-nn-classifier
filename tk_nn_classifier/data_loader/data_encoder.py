'''utility functions to prepare data for model training and prediction'''

import re
import numpy as np
from easy_tokenizer.normalizer import normalize_chars
from .word_vector import WordVector, maxabs

CONTEXT_SEPARATOR = "\t"


def file_itt(data_file):
    '''read each line of the file, generate the line '''
    with open(data_file, mode="r", encoding="utf-8") as data_fh:
        for line in data_fh:
            if not line.isspace():
                yield line.rstrip("\n")


class DataEncoder:
    '''
    data reader class to prepare the data in the expected format
    '''
    def __init__(
            self,
            embedding,
            token_encoding="max_embedding",
            data_format="train_mode",
            tokenizer_regexp=None
        ):
        """Reader to read data from either training/test files or from server
        protocol.

        Args:
        embedding: embedding file
        token_encoding: 'max_embedding': each token is a string with space
                                         separated words, represented as absmax
                                         of word embeddings
        data_format: format of the input data
                     "train_file": training files with label and original token
                     "server": server protocol without label and original token
        """

        self.embedding = WordVector(embedding)
        self.token_encoding = token_encoding
        if self.token_encoding not in [ \
            "word_id", "word_embedding", "max_embedding", \
            "subword_max_embedding", "subword_max_line_embedding", \
                "line_embedding", "subword_line_embedding"]:
            raise Exception(
                "Unknown token encoding '{}'".format(token_encoding))
        self.tokenizer_regexp = tokenizer_regexp or re.compile(r'\w+|[^\w\s]+')
        self.data_format = data_format
        self.has_token_regexp = re.compile(r'\w')
        if self.data_format not in ["train_mode", "service_mode"]:
            raise Exception("Unknown data format '{}'".format(data_format))

    def encode_tokens(self, string):
        '''
        tokenize the input string, and encode it to a embedding representation
        '''
        encoded_string = self.embedding.unk_vector
        if re.search(self.has_token_regexp, string):
            words = [
                match.group().upper()
                for match in self.tokenizer_regexp.finditer(string)
                if re.search(self.has_token_regexp, match.group())
            ]
            if words:
                embeddings = np.array(
                    list(self.embedding.get_vector(word) for word in words))
                encoded_string = maxabs(embeddings)

        return encoded_string


    def read_file(self, file):
        """Read data files and return the input for the model.

        Args:
        file: a single file_path
        """
        data = []
        labels = []
        lines = []
        for line in file_itt(file):
            label, encoded_input = self.read_line(line)
            data.append(encoded_input)
            labels.append(label)
            lines.append(line)
        return lines, np.array(data), np.array(labels)

    def read_line(self, line):
        '''
        read one line:

        params:
            - line: a string

        output: an encoded embedding to represent this line
        '''
        line = normalize_chars(line)
        if self.data_format == "train_mode":
            label, left_context, center_words, right_context, _, *features = \
                line.split(CONTEXT_SEPARATOR)
        elif self.data_format == "service_mode":
            left_context, center_words, right_context, *features = \
                line.split(CONTEXT_SEPARATOR)
            label = None

        if self.token_encoding == 'max_embedding':
            encoded_left_context = self.encode_tokens(left_context)
            encoded_right_context = self.encode_tokens(right_context)
            encoded_center_words = self.encode_tokens(center_words)
            if features:
                encoded_input = np.reshape(np.array([
                    encoded_left_context,
                    encoded_center_words,
                    encoded_right_context,
                    features
                ]), (-1))

            else:
                encoded_input = np.reshape(np.array([
                    encoded_left_context,
                    encoded_center_words,
                    encoded_right_context,
                ]), (-1))
        else:
            raise Exception(
                "un-support token encoding type'{}'".format(self.token_encoding))

        return label, encoded_input

    def _encode_match(self, match):
        """
        Encode the content of one match

        params:
            - match: a MatchedPhrase object

        output: an encoded embedding to represent this match
        """

        left_context = normalize_chars(match.left_context)
        center_words = normalize_chars(match.surface_form)
        right_context = normalize_chars(match.right_context)

        encoded_left_context = self.encode_tokens(left_context)
        encoded_center_words = self.encode_tokens(center_words)
        encoded_right_context = self.encode_tokens(right_context)

        encoded_input = np.reshape(
            np.array([
                encoded_left_context,
                encoded_center_words,
                encoded_right_context,
            ]), (-1))

        return encoded_input

    def read_match(self, match):
        '''
        Read one match

        params:
            - match: a MatchedPhrase object

        output: an encoded embedding to represent this match
        '''

        encoded_input = self._encode_match(match)
        return np.array([encoded_input])

    def read_matches(self, matches):
        '''
        Read multiple matches

        params:
            - matches: a list of MatchedPhrase objects

        output: an encoded embedding to represent these matches
        '''

        encoded_inputs = [self._encode_match(match) for match in matches]
        return np.array(encoded_inputs)
