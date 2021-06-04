# #!/usr/bin/env python
# # coding: utf-8
#
# # In[7]:
#
#
# import allennlp_models.pretrained
# from allennlp_models.pretrained import load_predictor
# from allennlp.data.tokenizers import Token
#
#
# # In[ ]:
#
#
# predictor = load_predictor("tagging-elmo-crf-tagger")
#
#
# # In[2]:
#
#
# sentence = "Jobs and Wozniak cofounded Apple in 1976."
# preds = predictor.predict(sentence)
# for p in preds:
#     print(p)
# # for word, tag in zip(preds["words"], preds["tags"]):
# #     print(word, tag)
#
#
# # In[9]:
#
#
# print(preds["logits"])
#
#
# # In[9]:
#
#
# preds["words"]
# t = Token(preds["words"][0])
# print(t.idx)
#
#
# # In[14]:
#
#
# models = allennlp_models.pretrained.get_pretrained_models()
# for m in models:
#     print(f"{models}\n")
#
#
# # In[15]:
#
#
# predictor = load_predictor("tagging-fine-grained-transformer-crf-tagger")
#
#
# # In[27]:
#
#
# sentence = "Michael Jordan is a professor at Berkeley. He    studied computer science at Goldman School with Spike Lee in  August 2014."
# preds = predictor.predict(sentence)
# # print(preds)
# for word, tag in zip(preds["words"], preds["tags"]):
#     print(word, tag, len(word)+word.count(" "))
#
#
# # In[33]:
#
#
# text = "Michael     Jordan is a professor at Berkeley. He    studied computer science at Goldman School with Spike Lee in  August 2014."
# offset = 16
# tokens = text.split()
# token_offsets = []
# curr_offset = 0
# for token in tokens:
#     token_offsets.append(curr_offset)
#     curr_offset += len(token) + 1  # the next token will be found len(token) + 1 chars later than this one
# print(token_offsets)
#
# joined = ' '.join(tokens)
# print(joined)
#
#
# # In[40]:
#
#
# # In[ ]:
#
#
#
#
