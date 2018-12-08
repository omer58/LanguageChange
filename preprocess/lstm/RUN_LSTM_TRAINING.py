import json
import lstm_model
import os.path
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import time
import torch.utils.data as td
import torch
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append('../model_factory')
import TokenCleaner
torch.backends.cudnn.benchmark=True
from torch.utils.data import DataLoader


DEFAULT_Q_FILE_PATH = '../../data_sets/qanta.dev.2018.04.18.json'
DEFAULT_Q_YEAR_PATH = '../../data_sets/wiki_article_to_year.pickle'
DEFAULT_W2YVD_PATH   = '../../data_sets/w2yv_dic.pickle'
DEFAULT_W2YVV_PATH   = '../../data_sets/w2yv_vals.npy'

DEFAULT_V_FILE_PATH = '../../data_sets/qanta.test.2018.04.18.json'
BATCH_SIZE      = 128
MAX_LENGTH      = 128
EMBEDDING_DIM   = 1019
NUM_EPOCHS      = 10

DEFAULT_YEAR_VEC= [0.0]*EMBEDDING_DIM


class YVDataset(td.Dataset):
  def __init__(self, file_path, wiki_year_dict, w2yv_dict, w2yv_vals):
    self.w2yvVals = w2yv_vals
    cleaner = TokenCleaner.Cleaner()
    print('Loading dataset from ', file_path)
    self.data_x, self.data_y = [], []
    with open(file_path,'r') as F:
        for thing in F:
            j = json.loads(thing)['questions']
            for question_chunk in j:
                wiki_page = question_chunk['page']
                if wiki_page in wiki_year_dict:
                    wiki_year = wiki_year_dict[wiki_page]
                    question_words = cleaner.clean(question_chunk['text'])
                    question_words = [w2yv_dict[word] for word in question_words if word in w2yv_dict]
                    sent_len = len(question_words)
                    question = question_words+([-1]*(MAX_LENGTH - sent_len)) if sent_len < MAX_LENGTH else question_words[-MAX_LENGTH:] #PAD or CONCAT
                    if len(question)!= MAX_LENGTH:
                        print('FEAT',len(question))
                        print('SENT',sent_len)
                        print(question_chunk['text'])
                    self.data_x.append(question)
                    self.data_y.append(wiki_year - 1000)

  def __getitem__(self, index):
    sentence = self.data_x[index]
    features = torch.FloatTensor([self.w2yvVals[word] if word != -1 else DEFAULT_YEAR_VEC for word in sentence])
    labels = torch.LongTensor([self.data_y[index]])
    return (features, labels)

  def __len__(self):
    return len(self.data_x)

class LSTM_Loader:
    def TIME(self):
        s = str(time.time()-self.sT).split('.')[0] + ' sec'
        self.sT=time.time()
        return s

    def __init__(self, name, word2yeardic_path=DEFAULT_W2YVD_PATH, word2yearval_path=DEFAULT_W2YVV_PATH, question_file_path=DEFAULT_Q_FILE_PATH, validate_file_path=DEFAULT_V_FILE_PATH, question_year_path=DEFAULT_Q_YEAR_PATH):
        self.sT = time.time()
        self.name = str(name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.q_file_path = question_file_path
        self.v_file_path = validate_file_path
        self.q_year_path = question_year_path
        self.w2yvD_path = word2yeardic_path
        self.w2yvV_path = word2yearval_path
        self.wiki_year_dict = pickle.load(open(self.q_year_path, 'rb'))

        if os.path.isfile('../../models/'+name) and False:
            self.lstm = torch.load('../../models/'+name)
        else:
            print('loading year dict', self.TIME())
            self.w2yv_dict = pickle.load(open(self.w2yvD_path, 'rb'))
            self.w2yv_vals = np.load(open(self.w2yvV_path, 'rb'))
            print('initializing model', self.TIME())
            self.lstm           = lstm_model.YearLSTM(EMBEDDING_DIM, BATCH_SIZE, MAX_LENGTH, self.device )
            self.lstm.to(self.device)
            self.loss_function  = nn.NLLLoss().to(self.device)
            self.optimizer      = optim.SGD(self.lstm.parameters(), lr=0.003, nesterov=True, momentum=0.9)
            print('preparing train structures', self.TIME())
            trainL, testL = self.prepare_dataloaders()
            results = self._train_all_epochs_(trainL,testL)
            self.save_data_model(results, self.lstm)


    def save_data_model(self,results, model):
        trnA, trnL, tstA, tstL = results
        fig, (ax1,ax2) = plt.subplots( nrows=1, ncols=2 )
        ax1.plot(trnA, label='train acc')                   #Plot 1
        ax1.plot(tstA, label='test  acc')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax2.plot(trnL, label='train loss')                  #Plot 2
        ax2.plot(tstL, label='test  loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        fig.savefig('../../data_sets/results/'+self.name)   # save the figure to file
        plt.close(fig)    # close the figure
        torch.save(model, '../../models/'+self.name)

    def _train_all_epochs_(self, training_data, validation=[], num_epochs=NUM_EPOCHS):
        print('training', self.TIME(), '\r',end='')
        train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
        print('Epoch 0')#, end='')
        len_data = len(training_data)
        for epoch in range(num_epochs):
            epoch_correct, epoch_loss, valid_correct, valid_loss = 0.0, 0.0, 0.0, 0.0
            for iii, (sentence, tag) in enumerate(training_data):
                print('\rdata', str(iii), len_data, int(time.time()-self.sT), 'sec', end='')
                self.lstm.zero_grad()
                self.lstm.hidden = self.lstm.init_hidden()
                tag = tag.view(-1)
                target = Variable(tag).to(self.device)
                sentence = Variable(sentence).to(self.device)
                pred_year = self.lstm(sentence) #[BATCH x 1019]
                target = target.view(-1)
                loss = self.loss_function(pred_year, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
                for i, batch_guess in enumerate(pred_year):
                    #print('i ', i)
                    #print('len batch: ', len(pred_year))
                    if abs(torch.argmax(batch_guess) - target[i]) <10:
                        epoch_correct +=1.0
                        #print('corrects: ', epoch_correct, 'batch_size ', BATCH_SIZE*len(training_data))
            train_accuracy.append(epoch_correct/BATCH_SIZE/len(training_data))
            train_loss.append(epoch_loss.item()/BATCH_SIZE/len(training_data))

            if validation:
                with torch.no_grad():
                    for iii, (sentence, tag) in enumerate(validation):
                        self.lstm.hidden = self.lstm.init_hidden()
                        tag = tag.view(-1)
                        target = Variable(tag).to(self.device)
                        sentence = Variable(sentence).to(self.device)
                        pred_year = self.lstm(sentence) #[BATCH x 1019]

                        loss = self.loss_function(pred_year, target)
                        valid_loss += loss
                        for i, batch_guess in enumerate(pred_year):
                            if abs(torch.argmax(batch_guess) - target[i]) < 10:
                                valid_correct +=1
                    test_accuracy.append(valid_correct/BATCH_SIZE/len(validation))
                    test_loss.append(valid_loss.item()/BATCH_SIZE/len(validation))
            print('Epoch',str(epoch), self.TIME(),' train_accuracy', train_accuracy[-1], ', train_loss', train_loss[-1],', test_accuracy', test_accuracy[-1],', test_loss', test_loss[-1])#, '\r', end='')
        return (train_accuracy, train_loss, test_accuracy, test_loss)

    def prepare_dataloaders(self):
        training_set, validation_set = YVDataset(self.q_file_path, self.wiki_year_dict, self.w2yv_dict, self.w2yv_vals), YVDataset(self.v_file_path, self.wiki_year_dict, self.w2yv_dict, self.w2yv_vals)
        trainloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(self.TIME())
        validloader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(self.TIME())
        return trainloader, validloader



INSTANCE = LSTM_Loader('TEST_1')
