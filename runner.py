from allennlp.data import Instance

from predictors.raw_text_predictor import AllenNLPRawTextPredictor
from readers.raw_text_reader import AllenNLPRawTextReader
import allennlp_models.tagging as tagging


article = "/Users/safr/Development/socom-sibr/allennlp_ner/data/article.txt"

raw_reader = AllenNLPRawTextReader()
instance = Instance(list(raw_reader.read(file_path=article))[0])
model = tagging.CrfTagger.from_archive(archive_file="https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2021-02-11.tar.gz")
predictor = AllenNLPRawTextPredictor(model=model, dataset_reader=raw_reader, frozen=False)
output = predictor.predict_instance(instance=instance)
print(output)
