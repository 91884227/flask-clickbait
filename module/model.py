#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[1]:


import torch
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda:0"


# # import self-define tool

# In[2]:


DEVICE = "cuda:0"

from data_preprocessing import encode

from module.network_structure import LSTM_model, LSTM_model_BI, GRU_model, GRU_model_BI

model = torch.load("ID_0034_0.12.ptc")

model.to(DEVICE)


# In[7]:


# test = encode("校園裝冷氣蘇揆搶功? 侯友宜:孩子們受惠最重要")

# input_ = torch.FloatTensor(test).unsqueeze(1).unsqueeze(0)

# model(input_.to(DEVICE))


# In[14]:


def ABS(input_: str)-> float:
    after_encode = encode(input_)
    after_tensor = torch.FloatTensor(after_encode).unsqueeze(1).unsqueeze(0)
    
    score = model(after_tensor.to(DEVICE))
    a, b = score.detach().cpu().squeeze()
    return(abs( (a-b).item() ))


# In[15]:


# ABS("校園裝冷氣蘇揆搶功? 侯友宜:孩子們受惠最重要 我去你的")


# In[ ]:


if __name__ == "__main__":
    buf = ABS("校園裝冷氣蘇揆搶功? 侯友宜:孩子們受惠最重要 我去你的")
    print(buf)

