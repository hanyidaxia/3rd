# import sys
# import os
# import numpy as np
# sys.path.append("../")
# import codes.models
# from excute import *
# from data_reader import *
# import torch
# from transformers import AlbertConfig, AlbertModel, AdamW, AlbertTokenizer, AlbertForMaskedLM
# from sklearn.metrics import precision_recall_fscore_support
# import json

# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# print(args)
# data_dir = '/home/jade/3rd/data/'
#
# processor = jade_processor(AlbertTokenizer.from_pretrained('albert-base-v2'), args.max_seq_length)
# reader = jade_reader(args.batch_size)
# train_examples = reader.get_train_examples(data_dir)
# print(len(train_examples), type(train_examples), np.shape(train_examples))
#
#
# for i, example in enumerate(train_examples):
#     inputs, labels = processor.convert_examples_to_tensor(example)

# from torch.nn import CosineSimilarity as cs
# input1 = torch.randn(100, 128)
# input2 = torch.randn(100, 128)
# cos = cs(dim=1, eps=1e-6)
# output = cs(input1, input2)
# print(output)


from similarity import read_file

key_reader = read_file('/home/jade/Downloads/results_review_new.txt')
dic = key_reader.split_keys()
print(len(dic))
read_file.look_n(dic, 2)



# print(train_examples.text)
