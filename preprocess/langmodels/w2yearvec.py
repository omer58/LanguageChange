from collections import defaultdict
import pickle
from math import log
import numpy as np


DENOISING_CUTOFF = 5

def dump_w2yv(word2YearVec, file):

    for i in word2YearVec:
        file.write(i + "++||++" + str(word2YearVec[i]) + '\n')

def load_w2yv(file):
    word2YearVec = defaultdict(list)

    for i,l in enumerate(file):
        #print(l)
        key = l.split('++||++')[0]
        vals = []
        for ent in l.split('++||++')[1][1:-2].split(','):
            if ent:
                #print('ent', ent)
                vals.append(float(ent))

        word2YearVec[key] = vals

    return word2YearVec


if __name__ == "__main__":

    LEN_YEAR_VECTOR = 1019
    year_book = pickle.load(open('../../data_sets/year_book.pickle', 'rb'))
    word2YearVec = defaultdict(list)            # { word -> LIST(year_i -> REAL) }
    totalYearCounts = defaultdict(int)          # { year -> int_tot_wrd_count }
    wordsInYear = defaultdict(set)              # { word -> SET(year0, year1, year2) }

    for year, year_page in enumerate(year_book):
        for word, count in year_page.items():
            try:
                word2YearVec[word][year] = count
            except IndexError:
                for i in range(LEN_YEAR_VECTOR):
                    word2YearVec[word].append(0.0)
                word2YearVec[word][year] = count
            totalYearCounts[year] += count
            wordsInYear[word].add(year)

    word2YearVec = dict(word2YearVec)

    print('normalizing...')
    '''WINDOW_SIZE = 5;
    half_window = int( WINDOW_SIZE/2)
    for word in word2YearVec:
        sum_ = 0
        for i in range(len(word2YearVec[word])):
            if i - half_window > -1 and i+half_window < len(word2YearVec[word]):
                tot = sum(word2YearVec[word][i - half_window : i + half_window +1])
                if tot == word2YearVec[word][i] and tot < DENOISING_CUTOFF:
                    word2YearVec[word][i] = 0

            elif i - WINDOW_SIZE > -1:
                tot = sum(word2YearVec[word][i-WINDOW_SIZE:i])
                if tot == word2YearVec[word][i] and tot < DENOISING_CUTOFF:
                    word2YearVec[word][i] = 0

            elif i+ WINDOW_SIZE <= len(word2YearVec[word]):

                tot = sum(word2YearVec[word][i:i+WINDOW_SIZE])
                if  tot == word2YearVec[word][i] and tot < DENOISING_CUTOFF:
                    word2YearVec[word][i] = 0


            sum_ += word2YearVec[word][i]
        '''

    for word in word2YearVec:
        sum_ = sum(word2YearVec[word])
        for i in range(len(word2YearVec[word])):
            word2YearVec[word][i] /= (1.0 * sum_)


    #scale word vectors, divide each word in year, by total occurance count of that word.
    '''
    print('tf-idfing...')
    for word, wordVec in word2YearVec.items():
        yearsWithWord = len(wordsInYear[word])

        for year, value in enumerate(wordVec):
            if value:
                try:
                    word2YearVec[word][year] = (value / totalYearCounts[year]) \
                                            * log(LEN_YEAR_VECTOR / yearsWithWord)

                except ZeroDivisionError:
                    print(word)
                    print(wordVec)
                    print(yearsWithWord)
                    print(value)
                    print(totalYearCounts[year])
                    print()
    '''



    map = {x:i for i, x in enumerate(word2YearVec.keys())}
    vecs = np.asarray(word2YearVec.items())
    print('pickling')
    pickle.dump(map, open('w2yv_dict_normalized.pickle', 'wb'))
    print('npy saving')
    np.save(open('w2yv_vals_normalized.npy', 'wb'), vecs)
