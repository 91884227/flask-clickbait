#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[3]:


import pandas as pd
import json
import numpy as np
import itertools
import collections
import tensorflow as tf
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import os
from datetime import date
from IPython.display import clear_output
clear_output()
from tqdm import tqdm
import sys
import pickle
from typing import List


# # setting

# In[4]:


# WORD_TO_WEIGHT = "dictionary.txt"
# LIMIT = 1
CUDA_VISIBLE_DEVICES = "0"
GPU_MEMORY_FRACTION = 0.7


# In[5]:


print("set GPU stat...")
cfg = tf.ConfigProto()
cfg.gpu_options.per_process_gpu_memory_fraction =  GPU_MEMORY_FRACTION ###設定gpu使用量
session = tf.Session(config = cfg)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES ###設定gpu編號


# In[6]:


# data_utils.download_data_gdown("./")


# In[7]:


print("prepare ws pos ner")
ws = WS("./data")
pos = POS("./data")
ner = NER("./data")
clear_output()


# ## encoding function

# In[8]:


with open("Encoding_par.pkl", 'rb') as input:
    le = pickle.load(input)


# In[10]:


def encode(input_: str) -> List[int]:
    #### segmentation part ####
    word_sentence_list = ws(
        [input_],
        sentence_segmentation = True, # To consider delimiters
        # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
        # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
        # coerce_dictionary = dictionary2, # words in this dictionary are forced
    )

    pos_sentence_list = pos(word_sentence_list)
    word_sentence_list, pos_sentence_list = word_sentence_list[0], pos_sentence_list[0]

    #### replace tag ####
    replace_tag = ["Nb", "Neu","Nc","Nd", "Nf"]

    after_replace = [tag if tag in replace_tag else  word for word, tag in zip(word_sentence_list, pos_sentence_list)]

    #### encode part ####

    command_word_list = list(le.classes_)

    after_replace = [i if i in command_word_list else "<UNK>" for i in after_replace]

    MAX_LEN = 20
    after_len_selection = after_replace[ :MAX_LEN]

    after_encode = le.transform(after_len_selection)

    return( [int(i) for i in after_encode] )


# In[11]:


encode(["校園裝冷氣蘇揆搶功? 侯友宜:孩子們受惠最重要"])


# In[12]:


if __name__ == '__main__':
    buf = encode("校園裝冷氣蘇揆搶功? 侯友宜:孩子們受惠最重要")
    print(buf)

