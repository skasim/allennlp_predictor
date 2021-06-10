from typing import Dict, List
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, \
    ELMoTokenCharactersIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, SpacyTokenizer


@DatasetReader.register("allennlp_raw_text_reader")
class AllenNLPRawTextReader(DatasetReader):
    DatasetReader.default_implementation = "allennlp_raw_text_reader"

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self.tokenizer = SpacyTokenizer()
        self._token_indexers = token_indexers or {"elmo": ELMoTokenCharactersIndexer(),
                                                  "token_characters": TokenCharactersIndexer(min_padding_length=4),
                                                  "tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Instance:
        with open(file_path, "r") as f:
            text = f.read()
            tokens = self.tokenizer.tokenize(text)

            yield self.text_to_instance(tokens)

    @overrides
    def text_to_instance(self,
                         words: List[Token],
                         ) -> Instance:
        # wrap each token in the file with a token object
        tokens = TextField(tokens=words, token_indexers=self._token_indexers)

        fields: Dict[str, Field] = {}
        fields = {"tokens": tokens}
        return Instance(fields=fields)
