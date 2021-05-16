import re


def tokenize(doc):
    return re.findall(r'\b\w\w+\b', doc)
