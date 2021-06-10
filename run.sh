allennlp predict \
  --output-file run_output.json \
  --include-package tagging \
  --predictor allennlp_raw_text_predictor \
  --use-dataset-reader \
  --silent \
  /Users/safr/Desktop/allennlp/fine-grained-ner.2021-02-11/ \
  data/article.txt
