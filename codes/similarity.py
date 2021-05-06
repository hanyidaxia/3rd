import sys
import os
import operator
import re
import numpy as np
import json
import copy
from data_reader import all_def
from sklearn.metrics.pairwise import cosine_similarity as cs


class read_file:

    def __init__(self, file_name):
        self.file_name = file_name

    def import_vector(self):
        embedding_dict = {}
        with open(self.file_name, 'r', encoding='UTF-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:])
                embedding_dict[word] = vector
            print('successfully import the language model %s' % (self.file_name))
        return embedding_dict

    def import_lexicon(self):
        with open(self.file_name, "r") as f:
            if '.txt' in self.file_name:
                file = f.readlines()
                file = "".join(file)
                file = file.split('\n')
                print('you successfully read a txt file, there are %d lines contained in the txt file' % (len(file)))
            elif '.json' in self.file_name:
                file = json.load(f)
                print('you successfully read a json file, there are %d keys contained in the dict file' % (len(file)))
        return file

    def write_txt_fromdic(self, data):
        with open(self.file_name, "w", encoding="UTF-8") as f:
            for k, v in data.items():
                f.write(str(k) + str(v))
                f.write('\n')
                f.write('\n')
            print('Done')

    def write_json(self, dic):
        with open(self.file_name, "w") as f:
            json.dump(dic, f)
            print("Done")

    def look_n(dic, n):
        i = 0
        for k, v in dic.items():
            i += 1
            if i < n + 1:
                print(k, v)

    def _read_task(self):
        print('Reading file:', self.file_name)
        data = all_def.load_json(self.file_name)
        print('Read line:', len(data))
        print('Done')
        return data

    def reader(self, data_dir):
        data = self._read_task(os.path.join(data_dir, self.file_name))
        return data

    def _key_collector(self):
        names = [
            'nike air max', 'nike air vapormax', 'nike air max 95', 'nike air max 90', 'nike air max 270',
            'nike air force 1',
            'nike huarache', 'nike react', 'nike kyrie', 'nike lebron', 'nike kd', 'nike pg',
            'nike cortez', 'nike flight', 'nike flyknit', 'nike free', 'nike pegasus', 'nike presto', 'nike roshe one',
            'nike zoom', 'jordan retro', 'adidas nmd', 'adidas boost', 'adidas ultraboost', 'adidas originals',
            'adidas d rose',
            'adidas stan smith', 'adidas superstars', 'puma roma', 'puma suede', 'converse chunk taylor',
            'new balance classics',
            'ascis tiger', 'reebok classics', 'reebok zig', 'under armour curry'
        ]

        return names

    def split_keys(self):
        names = self._key_collector()
        all_key_feature_dic = {}
        with open(self.file_name, 'r') as f:
            reviews = f.readlines()
            reviews = "".join(reviews)
            reviews = reviews.split('\n')

            review_dic = {}

            for review in reviews:
                if "out of 5 stars" not in review:
                    if "===" in review:
                        p_name = review.split('===')[0]
                        if p_name not in review_dic:
                            review_dic[p_name] = [review.split('===')[-1]]
                    else:
                        review_dic[p_name].append(review)
            #         all_def.look_n(review_dic, 1)

            for k, v in review_dic.items():
                for name in names:
                    for sub_v in v:
                        if isinstance(sub_v, str):
                            if name in sub_v.lower():
                                if name not in all_key_feature_dic:
                                    all_key_feature_dic[name] = v
                                else:
                                    all_key_feature_dic[name].append(v)
            #         all_def.look_n(all_key_feature_dic, 1)

            for k, v in all_key_feature_dic.items():
                tmp = []
                for sub_v in v:
                    if len(sub_v) != 0 and sub_v[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        #                     print(sub_v[1:])
                        tmp.append(sub_v[1:])
                all_key_feature_dic[k] = tmp

            for k, v in all_key_feature_dic.items():
                v = [i for i in v if i]
                all_key_feature_dic[k] = v

        return all_key_feature_dic


class compare_similarity:

    def __init__(self, embedding_tool):
        self.embedding_tool = embedding_tool

    def similarity(self, file):
        """this function requires the input data to be in dictionary format"""
        compare_list = []
        final_results = {}
        for k, v in file.items():
            tmp_list = [k]
            tmp_list.extend(v)
            compare_list.append(tmp_list)

        for i in compare_list:
            simi_dic = {}
            each_list = copy.deepcopy(i)
            for j in i:
                simi_sum = 0
                for m in each_list:
                    if j == m:
                        continue
                    else:
                        if j in self.embedding_tool.keys() and m in self.embedding_tool.keys():
                            doc1 = self.embedding_tool[j]
                            doc2 = self.embedding_tool[m]
                        else:
                            continue
                        if len(doc1) == 300 and len(doc2) == 300:
                            simi = cs([doc1, doc2])[0][1]
                            simi_sum += simi
                        else:
                            continue
                simi_dic[j] = simi_sum

            for k in file.keys():
                final_results[k] = max(simi_dic.items(), key=operator.itemgetter(1))[0]

        return final_results


file_reader = read_file('/media/jade/yi_Data/Data/Algorithm_1/new_new_attribute_lexicon.json')
embedding_reader = read_file('/media/jade/yi_Data/Data/models/glove.840B.300d/glove.840B.300d.txt')
embedding_tool = embedding_reader.import_vector()
similarity_calculator = compare_similarity(embedding_tool)
simi_dic = similarity_calculator.similarity(file_reader.import_lexicon())
