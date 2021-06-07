from typing import Dict, List, Iterator
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, SpacyTokenizer


@DatasetReader.register("allennlp_raw_text_reader")
class AllenNLPRawTextReader(DatasetReader):
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

            yield self.text_to_instance(tokens)





    @overrides
    def text_to_instance(self,
                         words: List[Token],
                         ) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField(tokens=words, token_indexers=self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        return Instance(fields)
