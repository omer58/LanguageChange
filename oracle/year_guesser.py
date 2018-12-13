#pseudo code, really
import pickle
import numpy as np
import sys
import os
sys.path.insert(0, '../preprocess/model_factory/')
sys.path.insert(0, '../preprocess/langmodels')

from TokenCleaner import Cleaner
EPSILON = sys.float_info.epsilon

def echo(s):
    os.system("echo  ' " +str(s)+ " ' ")




class Year_Guesser:
    def avg_year(self, ques_vec):
        tot = sum(ques_vec)
        ques_vec = [x/tot for x in ques_vec]
        s = 0
        for i, j in enumerate(ques_vec):
            s += i*j
        return round(s)


    def weighted_median(self, ques_vec):
        left_weight = 0
        total       = sum(ques_vec)
        ques_vec    = [x/total for x in ques_vec]
        total       = sum(ques_vec) # sum of weights is one
        fulcrum     = total/2       # should be 0.5

        #assert abs(0.5 - fulcrum) < EPSILON
        #assert abs(1.0 - total ) < EPSILON

        for i, x in enumerate(ques_vec):
            left_weight = left_weight+x
            if left_weight > fulcrum:
                return i

    def top_n_avg(self, ques_vec, n=1):
        #maxs = np.array([(0,0)]*n)

        #for i,el in enumerate(ques_vec):
            #echo(str(i) + " " + str(el))
        #    for j,le in enumerate(maxs):
        #        echo(str(i) + " " + str(el) + " " + str(j) + " "  + str(le))
        #        if el > le[0]:
        #            maxs[j] = np.array([el,i]);
        #            break;

        new_vec = list(enumerate(ques_vec))
        new_vec = new_vec[500:]
        new_vec.sort(key = lambda x: x[1], reverse = True)

        echo(new_vec[:10])
        return 5#sum([le[1] for le in maxs])/ len(maxs)



    def __init__(self):
        #import os
        #os.system('ls ../../data_sets')
        self.year_vecs = np.load('../../data_sets/w2yv_vals_wordnormalized.npy')
        self.word_map = pickle.load(open('../../data_sets/w2yv_dict_wordnormalized.pickle', 'rb'))


    def guess(self,question):
        import os
        #question_vec =np.empty(1019)
        matrix = np.zeros((len(question), 1019))
        echo(question)
        for i, word in enumerate(question.split(' ')):
            try:
                matrix[i] = self.year_vecs[self.word_map[word]]
                echo(word)
                self.top_n_avg(matrix[i],n=20)
            except KeyError:
                continue
            #question_vec = np.add(question_vec, self.year_vecs[self.word_map[word]])
        question_vec = matrix.sum(axis=0)
        pred = self.top_n_avg(question_vec,n=20)
        echo(pred)
        return pred+1000

if __name__ == "__main__":

    import json

    ds = json.loads(open('../qanta-codalab/data/qanta.dev.2018.04.18.json').read())
    clean = Cleaner()
    for iii in range(10):
        print(ds['questions'][iii])
        q_t = clean.clean(ds['questions'][iii]['text'])
        oracle = Year_Guesser()
        print(oracle.guess(q_t))
