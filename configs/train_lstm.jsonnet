{
  dataset_reader: {
    type: 'conll_03_reader'
  },
  train_data_path: 'data/train.txt',
  validation_data_path: 'data/validation.txt',
  model: {
    type: 'ner_lstm',
    embedder: {
      token_embedders: {
        tokens: {
        type: 'embedding',
          pretrained_file: "/Users/safr/Desktop/allennlp/glove.6B.50d.txt.zip",
          embedding_dim: 50,
          trainable: false
        }
      }
    },
    encoder: {
      type: 'lstm',
      input_size: 50,
      hidden_size: 25,
      bidirectional: true
    }
  },
  data_loader: {
    batch_size: 10,
    shuffle: true
  },
  trainer: {
    num_epochs: 10,
    patience: 3,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
}