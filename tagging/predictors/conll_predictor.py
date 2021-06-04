from allennlp.data.fields import TextField
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.tokenizers import Token


@Predictor.register("conll_03_predictor")
class CoNLL03Predictor(Predictor):

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')

        # outputs['tokens'] = [str(token) for token in instance.fields['tokens'].tokens]
        tokens = []
        print("****")
        tf = instance.fields["tokens"].tokens
        print(tf)
        for token in instance.fields['tokens'].tokens:
            t = token
            tokens.append({"token": t, "start": t.idx, "end": t.idx_end})
        outputs["tokens"] = tokens
        predicted = [label_vocab[l] for l in outputs['logits'].argmax(1)]
        # outputs['predicted'] = [label_vocab[l] for l in outputs['logits'].argmax(1)]
        # outputs['labels'] = instance.fields['label'].labels

        print(len(predicted))
        print(len(outputs["tokens"]))
        assert len(predicted) == len(outputs["tokens"])

        for i in range(0, len(predicted)):
            outputs["tokens"][i]["tag"] = predicted[i]
        print(outputs)
        return sanitize(outputs["tokens"])
