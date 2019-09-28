HAS_TOKEN_REGEXP = re.compile(r'\w')
TOKEN_REGEXP = re.compile(r'\w+|[^\w\s]+')

def tokenize(string):
    string = char_normalization(string)
    tokens = []
    if re.search(HAS_TOKEN_REGEXP, string):
        tokens = [
            match.group().upper()
            for match in TOKEN_REGEXP.finditer(string)
            if re.search(HAS_TOKEN_REGEXP, match.group())
        ]
    return tokens
