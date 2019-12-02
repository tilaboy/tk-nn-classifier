import re
from easy_tokenizer.normalizer import normalize_chars
from easy_tokenizer.tokenizer import Tokenizer
from easy_tokenizer.patterns import Patterns

HAS_TOKEN_REGEXP = re.compile(r'\w')
TOKEN_REGEXP = re.compile(r'\w+|[^\w\s]+')
TOKENIZER = Tokenizer()


def tokenize(string):
    '''tokenize string, and return the list of normalized tokens'''
    string = normalize_chars(string)
    try:
        TOKENIZER  # NOQA
    except NameError:
        TOKENIZER = Tokenizer()

    return [_norm_token(token) for token in TOKENIZER.tokenize(string)]


def _norm_token(token):
    norm_token = token.upper()
    if Patterns.URL_RE.fullmatch(token):
        norm_token = 'xxURLxx'
    elif Patterns.EMAIL_RE.fullmatch(token):
        norm_token = 'xxEMAILxx'
    elif Patterns.DOMAIN_RE.fullmatch(token):
        norm_token = 'xxAT_DOMAINxx'
    elif Patterns.YEAR_RE.fullmatch(token):
        norm_token = 'xxYEARxx'
    elif Patterns.EMAIL_RE.fullmatch(token):
        norm_token = 'xxEMAILxx'

    norm_token = Patterns.DIGIT_RE.sub('1', norm_token)
    return norm_token
