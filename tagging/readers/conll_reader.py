from typing import Dict, List, Iterator
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, SpacyTokenizer

import itertools


def is_divider(line):
    return line.strip() == ''


@DatasetReader.register("conll_03_reader")
class CoNLL03DatasetReader(DatasetReader):
    # need to register this as a conll_03_reader even though this is a text reader
    # todo figure out how to register two readers
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self.tokenizer = SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, "r") as f:
            text = f.read()
            tokens = self.tokenizer.tokenize(text)
            # text_field = TextField(self.tokenizer.tokenize(text),
            #                        self.token_indexers)
            # fields = {'text': text_field}
            # yield Instance(fields)

            yield self.text_to_instance(tokens)





    @overrides
    def text_to_instance(self,
                         words: List[Token],
                         ner_tags: List[str] = None
                         ) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField(tokens=words, token_indexers=self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        if ner_tags:
            fields['label'] = LabelField(ner_tags)
        return Instance(fields)
