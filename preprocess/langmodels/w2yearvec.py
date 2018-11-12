from collections import defaultdict
import pickle
from math import log

LEN_YEAR_VECTOR = 1019
year_book = pickle.load(open('../../data_sets/year_book.pickle', 'rb'))
word2YearVec = defaultdict(list)            # { word -> LIST(year_i -> REAL) }
totalYearCounts = defaultdict(int)          # { year -> int_tot_wrd_count }
wordsInYear = defaultdict(set)            # { word -> SET(year0, year1, year2) }

#make word vectors
for year, documents in enumerate(year_book):
    for doc in documents:
        for word in doc:
            try:
                word2YearVec[word][year] += 1
            except IndexError:
                for i in range(LEN_YEAR_VECTOR):
                    word2YearVec[word].append(0.0)
                word2YearVec[word][year] += 1
            totalYearCounts[year] += 1
            wordsInYear[word].add(year)

totalYears = len(year_book)

#scale word vectors, divide each word in year, by total occurance count of that word.
for word, wordVec in word2YearVec.items():
    yearsWithWord = len(wordsInYear[word])

    for year, value in enumerate(wordVec):
        if value:
            try:
                word2YearVec[word][year] = (value / totalYearCounts[year]) \
                                        * log(totalYears / yearsWithWord)

            except ZeroDivisionError:
                print(word)
                print(wordVec)
                print(yearsWithWord)
                print(value)
                print(totalYearCounts[year])
                print()


pickle.dump(word2YearVec, open('../../data_sets/w2yv.pickle', 'wb'))
