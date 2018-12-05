import json
import lstm_module
import os.path
import torch
import pickle

DEFAULT_Q_FILE_PATH = '../../../qanta-codalab/data/qanta.train.2018.04.18.json'
DEFAULT_Q_YEAR_PATH = '../../data_sets/wiki_article_to_year.pickle'
DEFAULT_W2YV_PATH   = '../../data_sets/w2yv.pickle'
DEFAULT_V_FILE_PATH = '../../../qanta-codalab/data/qanta.test.2018.04.18.json'

class LSTM_Loader:

    def __init__(name, word2yearvec_path=DEFAULT_W2YV_PATH, question_file_path=DEFAULT_Q_FILE_PATH, validate_file_path=DEFAULT_V_FILE_PATH, question_year_path=DEFAULT_Q_YEAR_PATH):
        self.name = str(name)
        self.q_file_path = question_file_path
        self.v_file_path = validate_file_path
        self.q_year_path = question_year_path
        self.w2yv_path = word2yearvec_path

        if os.path.isfile('../../models/'+name):
            self.lstm = torch.load('../../models/'+name)
        else:
            self.lstm = lstm_module.LSTM(self.w2yv_dict, self.name)
            self.train()


    def train(self):
        wiki_year_dict = pickle.load(open(self.w2yv_path, 'rb'))
        training_set, validation_set = [], []
        with open(self.q_file_path,'r') as F:
            for thing in F:
                j = json.loads(thing)['questions']
                for question_chunk in j:
                    question = question_chunk['text']
                    wiki_page = question_chunk['page']
                    if wiki_page in wiki_year:
                        wiki_year = wiki_year_dict[wiki_page]
                        training_set.append((question, wiki_year))
                    else:
                        continue #SKIP THIS QUESTION
        with open(self.v_file_path,'r') as F:
            for thing in F:
                j = json.loads(thing)['questions']
                for question_chunk in j:
                    question = question_chunk['text']
                    wiki_page = question_chunk['page']
                    if wiki_page in wiki_year:
                        wiki_year = wiki_year_dict[wiki_page]
                        validation_set.append((question, wiki_year))
                    else:
                        continue #SKIP THIS QUESTION
        self.lstm.train(training_data, validation)
