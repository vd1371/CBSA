from transformers import AutoModel
import torch
import torch.nn as nn


from ._initialize_bert import initialize_bert

import torch

class FineTune(nn.Module):
		
    def __init__(self, bert):
      
      super(FineTune, self).__init__()

      self.bert = bert 
      self.dropout = nn.Dropout(0.1)     
      self.relu =  nn.ReLU()
      self.fc1 = nn.Linear(768,512)     
      self.fc2 = nn.Linear(512,1)
      self.output = nn.Sigmoid()

    def forward(self, sent_id, mask):

      #pass the inputs to the model
      _, cls_hs = self.bert(sent_id, attention_mask = mask, return_dict=False)

      X = self.fc1(cls_hs)

      X = self.relu(X)

      X = self.dropout(X)

      # output layer
      X = self.fc2(X)
      
      # apply softmax activation
      X = self.output(X)

      return X