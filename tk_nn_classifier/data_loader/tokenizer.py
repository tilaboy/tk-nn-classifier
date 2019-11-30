import re
from easy_tokenizer.normalizer import normalize_chars
from easy_tokenizer.tokenizer import Tokenizer

HAS_TOKEN_REGEXP = re.compile(r'\w')
TOKEN_REGEXP = re.compile(r'\w+|[^\w\s]+')
TOKENIZER = Tokenizer()

def tokenize(string):
    string = normalize_chars(string.upper())
    try:
        TOKENIZER
    except NameError:
        TOKENIZER = Tokenizer()

    return list(TOKENIZER.tokenize(string))
