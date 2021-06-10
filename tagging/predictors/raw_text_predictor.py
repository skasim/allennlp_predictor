from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Token
from overrides import overrides
import numpy as np


@Predictor.register("allennlp_raw_text_predictor")
class AllenNLPRawTextPredictor(Predictor):
    Predictor.default_implementation = "allennlp_raw_text_predictor"

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels') # gets labels {0: 'O', 1: 'I-ORG', 2: 'U-GPE', 3: 'B-ORG', 4: 'L-ORG', 5: 'B-DATE', 6: 'L-DATE', 7: 'U-PERSON', 8: 'B-PERSON', 9: 'L-PERSON', 10: 'I-DATE', 11: 'U-ORG', 12: 'U-CARDINAL', 13: 'U-NORP', 14: 'U-DATE', 15: 'I-MONEY', 16: 'I-PERSON', 17: 'B-MONEY', 18: 'L-MONEY', 19: 'B-GPE', 20: 'L-GPE', 21: 'B-PERCENT', 22: 'L-PERCENT', 23: 'U-ORDINAL', 24: 'I-WORK_OF_ART', 25: 'B-CARDINAL', 26: 'L-CARDINAL', 27: 'I-PERCENT', 28: 'I-GPE', 29: 'I-CARDINAL', 30: 'I-EVENT', 31: 'B-TIME', 32: 'L-TIME', 33: 'B-LOC', 34: 'L-LOC', 35: 'B-WORK_OF_ART', 36: 'L-WORK_OF_ART', 37: 'B-QUANTITY', 38: 'L-QUANTITY', 39: 'I-FAC', 40: 'I-TIME', 41: 'I-QUANTITY', 42: 'U-MONEY', 43: 'U-LOC', 44: 'B-FAC', 45: 'L-FAC', 46: 'B-EVENT', 47: 'L-EVENT', 48: 'I-LAW', 49: 'I-LOC', 50: 'U-PRODUCT', 51: 'U-TIME', 52: 'I-PRODUCT', 53: 'B-NORP', 54: 'L-NORP', 55: 'B-PRODUCT', 56: 'L-PRODUCT', 57: 'B-LAW', 58: 'L-LAW', 59: 'U-LANGUAGE', 60: 'U-WORK_OF_ART', 61: 'U-QUANTITY', 62: 'U-FAC', 63: 'U-EVENT', 64: 'I-NORP', 65: 'U-LAW', 66: 'B-ORDINAL', 67: 'L-ORDINAL', 68: 'I-ORDINAL', 69: 'B-LANGUAGE', 70: 'L-LANGUAGE', 71: 'U-PERCENT', 72: 'I-LANGUAGE'}

        # print(label_vocab)
        # print(outputs["logits"][0])
        # print(len(outputs["logits"][0]))
        # print((outputs["logits"][0]).argmax(0))
        #
        # indices_of_max_value = []
        # for i in outputs["logits"]:
        #     indices_of_max_value.append(i.argmax(0))
        #
        # print(indices_of_max_value)
        outputs['tokens'] = instance.fields['tokens'].tokens
        outputs['predicted'] = [label_vocab[l] for l in outputs['logits'].argmax(1)]  # gets the index for the max logit value and then finds the local_vocab value matching it
        assert len(outputs["predicted"]) == len(outputs["tokens"])


        santized = []
        for i in range(0, len(outputs["predicted"])):
            santized.append({
                "token": outputs["tokens"][i],
                "start": (outputs["tokens"][i]).idx,
                "end": (outputs["tokens"][i]).idx_end,
                "tag": outputs["predicted"][i]
            })

        print(outputs)
        return santized

        # tokens = []
        # for token in instance.fields['tokens'].tokens:
        #     t = token
        #     tokens.append({"token": t, "start": t.idx, "end": t.idx_end})
        # outputs["tokens"] = tokens
        # predicted = [label_vocab[l] for l in outputs['logits'].argmax(1)]

        # assert len(predicted) == len(outputs["tokens"])

        # for i in range(0, len(predicted)):
        #     outputs["tokens"][i]["tag"] = predicted[i]
        # print(sanitize(outputs["tokens"]))
        # return sanitize(outputs["tokens"])
