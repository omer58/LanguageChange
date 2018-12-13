import json
import os
import sys
sys.path.insert(0, '../preprocess/lstm')
import conv_model_simple as M
import os.path
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
#from matplotlib import pyplot as plt
import time
import torch.utils.data as td
import torch
import numpy as np
from torch.autograd import Variable
import sys

sys.path.insert(0, '../preprocess/model_factory')
import TokenCleaner
torch.backends.cudnn.benchmark=True
from torch.utils.data import DataLoader
from operator import itemgetter

#DEFAULT_Q_FILE_PATH = '../../data_sets/qanta.train.2018.04.18.json'
DEFAULT_Q_FILE_PATH = '../../../../qanta-codalab/data/qanta.dev.2018.04.18.json'
DEFAULT_Q_YEAR_PATH = '../../data_sets/wiki_article_to_year.pickle'
DEFAULT_W2YVD_PATH   = '../data_sets/w2yv_dic.pickle'
DEFAULT_W2YVV_PATH   = '../data_sets/w2yv_vals.npy'
#DEFAULT_V_FILE_PATH = '../../data_sets/qanta.test.2018.04.18.json'
DEFAULT_V_FILE_PATH =  '../../../../qanta-codalab/data/qanta.test.2018.04.18.json'

BATCH_SIZE      = 1
MAX_LENGTH      = 64
EMBEDDING_DIM   = 1019
NUM_EPOCHS      = 40

DEFAULT_YEAR_VEC= [0.0]*EMBEDDING_DIM


class nn_year_guesser:

    def __init__(self):
        print('LOADING MODEL FROM DISK')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.lstm = M.YearLSTM(EMBEDDING_DIM, BATCH_SIZE, MAX_LENGTH, self.device )

        self.lstm.load_state_dict(torch.load('../models/CONV_simple_adadelta_buckets_adadelta'))
        self.cleaner = TokenCleaner.Cleaner()
        self.w2yv_dict = pickle.load(open(DEFAULT_W2YVD_PATH, 'rb'))
        self.w2yvVals = np.load(open(DEFAULT_W2YVV_PATH, 'rb'))


    def run(self, question):


        question_words = self.cleaner.clean(question.lower())
        #self.questions.append([word for word in question_words if word in self.w2yv_dict])
        question_words = [self.w2yv_dict[word] for word in question_words if word in self.w2yv_dict]
        sent_len = len(question_words)
        question = question_words+([-1]*(MAX_LENGTH - sent_len)) if sent_len < MAX_LENGTH else question_words[-MAX_LENGTH:] #PAD or CONCAT
        if len(question)!= MAX_LENGTH:
            print('FEAT',len(question))
            print('SENT',sent_len)
            #print(question_chunk['text'])

        sentence = question


        features = torch.FloatTensor(np.asarray([self.w2yvVals[word]*100 if word != -1 else DEFAULT_YEAR_VEC for word in sentence]))

        sentence = features


        sentence = Variable(sentence).to(self.device)
        pred_year = self.lstm(sentence) #[BATCH x 1019]
        pred_year = pred_year.view(-1)

        for i, batch_guess in enumerate(pred_year):
            print(batch_guess)




guesser =  nn_year_guesser()
guesser.run('this is a question')
