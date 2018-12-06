import json
import lstm_model
import os.path
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import time
from torch.autograd import Variable
import sys
sys.path.append('../model_factory')
import TokenCleaner


DEFAULT_Q_FILE_PATH = '../../../../qanta-codalab/data/qanta.train.2018.04.18.json'
DEFAULT_Q_YEAR_PATH = '../../data_sets/wiki_article_to_year.pickle'
DEFAULT_W2YV_PATH   = '../../data_sets/w2yv.pickle'
DEFAULT_V_FILE_PATH = '../../../../qanta-codalab/data/qanta.test.2018.04.18.json'




class LSTM_Loader:
    def TIME(self):
        s = str(time.time()-self.sT).split('.')[0] + ' sec'
        self.sT=time.time()
        return s

    def __init__(self, name, word2yearvec_path=DEFAULT_W2YV_PATH, question_file_path=DEFAULT_Q_FILE_PATH, validate_file_path=DEFAULT_V_FILE_PATH, question_year_path=DEFAULT_Q_YEAR_PATH):
        self.sT = time.time()
        self.name = str(name)
        self.q_file_path = question_file_path
        self.v_file_path = validate_file_path
        self.q_year_path = question_year_path
        self.w2yv_path = word2yearvec_path
        self.wiki_year_dict = pickle.load(open(self.q_year_path, 'rb'))

        if os.path.isfile('../../models/'+name):
            self.lstm = torch.load('../../models/'+name)
        else:
            print('loading year dict', self.TIME())
            w2yv_dict = pickle.load(open(self.w2yv_path, 'rb'))
            print('initializing model', self.TIME())
            self.lstm           = lstm_model.YearLSTM(w2yv_dict )
            self.loss_function  = nn.NLLLoss()
            self.optimizer      = optim.SGD(self.lstm.parameters(), lr=0.003, nesterov=True, momentum=0.9)
            print('preparing train structures', self.TIME())
            self.train()


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

    def _train_all_epochs_(self, training_data, validation=[], num_epochs=50):
        print('training', self.TIME())
        train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
        print('Epoch 0', end='')
        for epoch in range(num_epochs):
            epoch_correct, epoch_loss, valid_correct, valid_loss = 0.0, 0.0, 0.0, 0.0
            for sentence, tag in training_data:
                tag = tag-1000
                self.lstm.zero_grad()
                self.lstm.hidden = self.lstm.init_hidden()
                print('YEAR', tag)
                target = Variable(torch.tensor([tag]))
                print(target)
                pred_year_softmax = self.lstm(sentence)
                pred_year = torch.argmax(pred_year_softmax)
                print(pred_year)
                print(pred_year_softmax)
                loss = self.loss_function(pred_year, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                if pred_year == target:
                    epoch_correct +=1
            train_accuracy.append(epoch_correct)
            train_loss.append(epoch_loss)

            if validation:
                with torch.no_grad():
                    for example_X, example_Y in validation:
                        pred_year  = self.lstm(inputs)
                        target = Variable(torch.tensor(example_Y))
                        v_loss = loss_function(pred_year, target)
                        if pred_year == target:
                            valid_correct +=1
                        valid_loss += loss
                    test_accuracy.append(valid_correct)
                    test_loss.append(valid_loss)
        print('Epoch',str(epoch), self.TIME(),'train_accuracy, train_loss, test_accuracy, test_loss', train_accuracy, train_loss, test_accuracy, test_loss, '\r', end='')
        return (train_accuracy, train_loss, test_accuracy, test_loss)

    def train(self):
        cleaner = TokenCleaner.Cleaner()
        training_set, validation_set = [], []
        with open(self.q_file_path,'r') as F:
            for thing in F:
                j = json.loads(thing)['questions']
                for question_chunk in j:
                    question = cleaner.clean(question_chunk['text'])
                    wiki_page = question_chunk['page']
                    if wiki_page in self.wiki_year_dict:
                        wiki_year = self.wiki_year_dict[wiki_page]
                        training_set.append((question, wiki_year))
                    else:
                        continue #SKIP THIS QUESTION
        with open(self.v_file_path,'r') as F:
            for thing in F:
                j = json.loads(thing)['questions']
                for question_chunk in j:
                    question = question_chunk['text']
                    wiki_page = question_chunk['page']
                    if wiki_page in self.wiki_year_dict:
                        wiki_year = self.wiki_year_dict[wiki_page]
                        validation_set.append((question, wiki_year))
                    else:
                        continue #SKIP THIS QUESTION

        results = self._train_all_epochs_(training_set, validation_set)
        self.save_data_model(results, self.lstm)
