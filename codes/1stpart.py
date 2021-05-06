import numpy as np
import sys
import json
import re
import spacy

nlp = spacy.load("en_core_web_sm")
from similarity import read_file
from similarity import compare_similarity

embedding_reader = read_file('/media/jade/yi_Data/Data/models/glove.840B.300d/glove.840B.300d.txt')
embedding_tool = embedding_reader.import_vector()
from sklearn.metrics.pairwise import cosine_similarity as cs


class pick_sentence:

    def __init__(self, hold):
        self.hold = hold

    def get_file(self, dic, attribute_dic, embedding_tool):
        key_dic = {}
        for k in attribute_dic.keys():
            for all_k, all_score in dic.items():
                doc = nlp(all_k)
                for token in doc:
                    if token.pos_ == 'NOUN':
                        embd_1 = embedding_tool[token.text]
                        embd_2 = embedding_tool[k]
                        if len(embd_1) == 300 and len(embd_2) == 300:
                            simi = cs([embd_2, embd_1])[0][1]
                            if simi >= self.hold:
                                for chunk in doc.noun_chunks:
                                    if token.text in chunk.text:
                                        if chunk.text not in key_dic:
                                            key_dic[chunk.text] = all_score
                                        else:
                                            key_dic[chunk.text].append(all_score)
                                    else:
                                        if token.text not in key_dic:
                                            key_dic[token.text] = all_score
                                        else:
                                            key_dic[token.text].append(all_score)
        return key_dic


all_text_reader = read_file('all.json')
all_text = all_text_reader.reader('data/')
file_reader = read_file('/media/jade/yi_Data/Data/Algorithm_1/new_new_attribute_lexicon.json')
attribute_dic = file_reader.import_lexicon()
similarity_calculator = compare_similarity(embedding_tool)
simi_dic = similarity_calculator.similarity(file_reader.import_lexicon())
sentence = pick_sentence(0.6)
result = sentence.get_file(all_text, attribute_dic, embedding_tool)
