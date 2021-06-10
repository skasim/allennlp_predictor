from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance


@Predictor.register("allennlp_raw_text_predictor")
class AllenNLPRawTextPredictor(Predictor):
    Predictor.default_implementation = "allennlp_raw_text_predictor"

    def predict_instance(self, instance: Instance) -> JsonDict:
        print(f"INSTANCE: {instance}")
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')

        tokens = []
        # tf = instance.fields["tokens"].tokens
        # print(tf)
        for token in instance.fields['tokens'].tokens:
            t = token
            tokens.append({"token": t, "start": t.idx, "end": t.idx_end})
        outputs["tokens"] = tokens
        predicted = [label_vocab[l] for l in outputs['logits'].argmax(1)]

        assert len(predicted) == len(outputs["tokens"])

        for i in range(0, len(predicted)):
            outputs["tokens"][i]["tag"] = predicted[i]
        print(outputs)
        return sanitize(outputs["tokens"])
