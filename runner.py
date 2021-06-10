from allennlp.data import DatasetReader, Vocabulary, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor

from predictors.raw_text_predictor import AllenNLPRawTextPredictor
from readers.raw_text_reader import AllenNLPRawTextReader
from allennlp.common.params import Params
import allennlp_models.pretrained as pretrained  # necessary to import models
from allennlp.common.util import import_module_and_submodules

cfg_file = "/Users/safr/Development/socom-sibr/allennlp_ner/configs/fine-grained/config.json"
article = "/Users/safr/Development/socom-sibr/allennlp_ner/data/article.txt"
model_dir = "/Users/safr/Desktop/open_model/fine-grained-ner.2021-02-11"
weights_file = "/Users/safr/Desktop/open_model/fine-grained-ner.2021-02-11/weights.th"
tagging_dir = "tagging"
# import_module_and_submodules(package_name=tagging_dir)

reader = DatasetReader()
print(Params.from_file(cfg_file).get("dataset_reader"))
reader.from_params(params=Params.from_file(cfg_file).get("dataset_reader"))
print(f"READERS AVAILABLE: {reader.list_available()}")
instance = reader.read(file_path=article)

# raw_reader = AllenNLPRawTextReader()
# print(f"READERS AVAILABLE: {raw_reader.list_available()}")
# instance = raw_reader.read(file_path=article)
# # TODO not sure why the instance object is a generator when passed in to this method
# for i in instance:
#     instance = Instance(i)

vocab = Vocabulary()
model = Model(vocab=vocab, serialization_dir=model_dir)

print(f"MODELS AVAILABLE: {model.list_available()}")
model.load(config=Params.from_file(cfg_file), serialization_dir=model_dir, weights_file=weights_file)

# predictor = AllenNLPRawTextPredictor(model=model, dataset_reader=raw_reader)
# predictor.by_name("allennlp_raw_text_predictor")

predictor = Predictor(model=model, dataset_reader=reader)
predictor.from_params(params=Params.from_file(cfg_file))
print(f"PREDICTORS AVAILABLE: {predictor.list_available()}")
outputs = predictor.predict_instance(instance=instance)


