allennlp public models from https://storage.googleapis.com/allennlp-public-models/

run predict with command
```
allennlp predict \
  --output-file $OUTPUT_FILE \
  --include-package tagging \
  --predictor conll_03_predictor \
  --use-dataset-reader \
  --silent \
  /Users/safr/Desktop/allennlp/fine-grained-ner.2021-02-11/ \
  data/article.txt
```
