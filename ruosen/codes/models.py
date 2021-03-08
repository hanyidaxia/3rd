import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import excute
from excute import *

print("import models file ..")


class albert(nn.Module):

    def __init__(self, bert, classes):
        super().__init__()
        self.bert = bert
        d_model = bert.embeddings.word_embeddings.weight.size(1)
        self.dense = nn.Linear(d_model, classes)
        # self.conv3= nn.Conv1d(512, 1024, kernel_size=3, stride = )

        self.fc1 = nn.Linear(764, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 4)

    def forward(self, inp, inp_mask):
        # encoder

        # print("********************************************" + "inp" +
        #         str(type(inp))+ str(np.shape(inp)) +
        #       "********************************************"  )

        output = self.bert(inp, inp_mask)[0]
        # print("1st output  " + str(np.shape(output)))
        #
        # output = F.relu(self.conv1(output))

        # print("2nd output  " + str(np.shape(output)))

        # output = F.relu(self.conv2(output))
        # print("3rd output  " + str(np.shape(output)))

        output = F.relu(self.fc1(output))
        # print("4th output  " + str(np.shape(output)))

        output = F.relu(self.fc2(output))
        # print("5th output  " + str(np.shape(output)))
        output = self.out(output)
        # print("final output  " + str(np.shape(output)))
        #
        # output = self.dense(output[:, 1:-1])
        # print("2nd output  " + str(np.shape(output)))

        # output = self.dropout(output)
        # output = F.relu(self.conv2(output))

        # print("********************************************" +
        #         str(type(output))+ str(np.shape(output)) +
        #       "********************************************"  )

        return output

    def init_parameters(self):
        for p in self.dense.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


print('import models file success')
