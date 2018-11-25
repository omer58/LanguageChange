#pseudo code, really
import pickle
import numpy as np
import sys
sys.path.insert(0, '../preprocess/model_factory/')

from TokenCleaner import Cleaner


class Year_Guesser:
    def avg_year(self, ques_vec):
        tot = sum(ques_vec)
        ques_vec = [x/tot for x in ques_vec]
        s = 0
        for i, j in enumerate(ques_vec):
            s += i*j 
        return round(s)



    def weighted_median(ques_vec):
        total=sum(ques_vec)
        ques_vec=[x/total for x in ques_vec]  #sum of weights is one
        #print (ques_vec)
        total=sum(ques_vec)

        fulcrum=total/2 #(should be 0.5)
        left_weight=0
        for i, x in enumerate(ques_vec):
            left_weight=left_weight+x
            #print (x)
            #print (left_weight)
            if left_weight>fulcrum:
                return i
            


    def __init__(self):
        self.year_vecs = pickle.load(open('../data_sets/w2yv_sample.pickle', 'rb'))


    def guess(self,question):
        question_vec =[0.0]*1019
        for word in question:
            if word in self.year_vecs:
                question_vec = np.add(question_vec, self.year_vecs[word])
        return self.avg_year(question_vec)+1000


import json

ds = json.loads(open('../qanta-codalab/data/qanta.dev.2018.04.18.json').read())
clean = Cleaner()
for iii in range(10):
    print(ds['questions'][iii])
    q_t = clean.clean(ds['questions'][iii]['text'])
    oracle = Year_Guesser()
    print(oracle.guess(q_t))
