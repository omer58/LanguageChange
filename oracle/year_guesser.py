#pseudo code, really
import pickle
import numpy as np

def class Year_Guesser:
    def avg_year(self, ques_vec):
        tot = sum(ques_vec)
        ques_vec = [x/tot for x in ques_vec]
        s = 0
        for i, j in enumerate(l):
            s += i*j
        return round(s)

    def __init__(self):
        self.year_vecs = pickle.load('../data_sets/y2wv.p')
    
    
    def guess(self,question):
        question_vec =[0.0]*2018
        for word in question:
            question_vec = np.add(question_vec, self.year_vecs[word])
        return self.avg_year(question_vec)


