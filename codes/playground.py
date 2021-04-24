import sys
import os
import numpy as np
sys.path.append("../")
import codes.models
from excute import *
from data_reader import *
from transformers import AlbertConfig, AlbertModel, AdamW, AlbertTokenizer, AlbertForMaskedLM
from sklearn.metrics import precision_recall_fscore_support
import json

# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# print(args)
data_dir = '/home/jade/3rd/data/'

processor = jade_processor(AlbertTokenizer.from_pretrained('albert-base-v2'), args.max_seq_length)
reader = jade_reader(args.batch_size)
train_examples = reader.get_train_examples(data_dir)
print(len(train_examples), type(train_examples), np.shape(train_examples))



for i, example in enumerate(train_examples):
    inputs, labels = processor.convert_examples_to_tensor(example)

# print(train_examples.text)
