import itertools
from typing import Iterator

from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer


def test_something():
    tokens = SpacyTokenizer().tokenize("My name is Joe I am writing from Greenland in 1776. But I visited India in 1984")
    print(tokens)

    for t in tokens:
        print(t.ent_type_)
        print(f"{t}, start:{t.idx} end: {t.idx_end}")


test_something()

def read_file(file_path="/Users/safr/Development/socom-sibr/allennlp_ner/data/article.txt"):
    with open(file_path, 'r') as f:
        return f.read()

# print(read_file())

def text_field():
    # t = "My name is Joe I am writing from Greenland in 1776"
    t = "Leeds NNP I-NP I-ORG\n"
    tokenizer = SpacyTokenizer()
    token_indexers = {"tokens": SingleIdTokenIndexer()}
    text_field = TextField(tokenizer.tokenize(t),
                           token_indexers)
    print(text_field)

# text_field()

def is_divider(line):
    return line.strip() == ''

def read_this(file_path: str) -> Iterator[Instance]:
    with open(file_path, 'r') as conll_file:
        # itertools.groupby is a powerful function that can group
        # successive items in a list by the returned function call.
        # In this case, we're calling it with `is_divider`, which returns
        # True if it's a blank line and False otherwise.
        tokenizer = SpacyTokenizer()
        for divider, lines in itertools.groupby(conll_file, is_divider):
            # skip over any dividing lines
            if divider: continue
            # # get the CoNLL fields, each token is a list of fields
            # fields = [l.strip().split() for l in lines]
            # # switch it so that each field is a list of tokens/labels
            # fields = [l for l in zip(*fields)]
            # only keep the tokens and NER labels

            tokens = [tokenizer.tokenize(l) for l in lines]
            print(f"tokens {tokens}")

            # tokens, _, _, ner_tags = fields

            # print(f"tokens {tokens} and ner-tags {ner_tags}")

            # yield self.text_to_instance(tokens, ner_tags)

# read_this("/Users/safr/Development/socom-sibr/allennlp_ner/data/test2.txt")