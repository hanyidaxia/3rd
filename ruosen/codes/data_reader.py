import sys

sys.path.append('../')
import torch
import json
import excute
from excute import *

"""
my codes requires all input file to be in json format
"""


class all_def:

    def write_file(file_name, data_set):
        with open(file_name, "w", encoding="UTF-8") as f:
            for i in data_set:
                f.write(str(i))
                f.write('\n')
            print('Done')

    def write_txt_fromdic(file_name, data):
        with open(file_name, "w", encoding="UTF-8") as f:
            for k, v in data.items():
                f.write(str(k) + str(v))
                f.write('\n')
                f.write('\n')
            print('Done')

    def write_json(file_name, dic):
        with open(file_name, "w") as f:
            json.dump(dic, f)
            print("Done")

    def look_n(dic, n):
        i = 0
        for k, v in dic.items():
            i += 1
            if i < n + 1:
                print(k, v)

    def load_json(file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
            print(type(data))
        return data

    def load_txt(file_name):
        with open(file_name, 'rb', encoding='UTF-8') as f:
            data = f.readlines()
        return data

    def test_dic(n, dic):
        tmp_dic = {}
        p = 0
        for k, v in dic.items():
            if p < n:
                if k not in tmp_dic:
                    tmp_dic[k] = v
                else:
                    tmp_dic[k].append(v)
                p += 1
        return tmp_dic


class get_examples:
    def __init__(self, text, label):
        self.text = text
        self.label = label


class DataReader(object):

    @classmethod
    def _read_task(cls, input_file):
        print('Reading file:', input_file, file=excute.SHELL_OUT_FILE, flush=True)
        data = all_def.load_json(input_file)
        print('Read line:', len(data), file=excute.SHELL_OUT_FILE, flush=True)
        print('Done', file=excute.SHELL_OUT_FILE, flush=True)
        return data


class jade_reader(DataReader):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def _create_examples(self, inputs):
        examples = []
        total_examples = len(inputs)
        seq = []
        labels = []
        cnter = 1
        for review, lab in inputs.items():
            cnter += 1
            if cnter % 1000 == 0:
                print("\rProcessed Examples: {}/{}".format(cnter, total_examples), end='\r', file=excute.SHELL_OUT_FILE,
                      flush=True)
            if cnter % self.batch_size == 0:
                seq.append(review)
                labels.append(int(lab))
                examples.append(get_examples(text=seq, label=labels))
                seq = []
                labels = []
        if len(seq):
            examples.append(get_examples(text=seq, label=labels))
        print("\rProcessed Examples: {}/{}".format(len(examples), total_examples), file=excute.SHELL_OUT_FILE,
              flush=True)

        return examples

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_task(os.path.join(data_dir, 'train.json')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_task(os.path.join(data_dir, 'dev.json')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_task(os.path.join(data_dir, 'test.json')))


class jade_processor(object):
    def __init__(self, tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def convert_examples_to_tensor(self, examples):
        examples.label = [int(i) for i in examples.label]
        labels = torch.tensor(examples.label).unsqueeze(0)
        inputs_ids = []
        inputs_mask = []
        token_type_ids = []

        for seq in examples.text:
            print(seq)
            inputs_raw = self.tokenizer(seq, max_length=self.max_seq_len, padding='max_length', truncation=True,
                                        add_special_tokens=True)
            print(inputs_raw)
            inputs_ids.append(torch.tensor(inputs_raw["input_ids"]).unsqueeze(0))
            inputs_mask.append(torch.tensor(inputs_raw["attention_mask"]).unsqueeze(0))
            token_type_ids.append(torch.tensor(inputs_raw["token_type_ids"]).unsqueeze(0))
        inputs = [inputs_ids, inputs_mask, token_type_ids]

        if excute.USE_CUDA:
            inputs = [i.cuda() for i in inputs]
            labels = labels.cuda()

        return inputs, labels

    def convert_tensor_to_tokens(self, tensor):
        ids = tensor.cpu().numpy().tolist()
        for i in range(len(ids)):
            ids[i] = self.tokenizer.decode(ids[i])
        return ids
