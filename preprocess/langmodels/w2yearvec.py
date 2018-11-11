from collections import defaultdict
import pickle

year_book = pickle.loads('../../data_sets/year_book_sample.pickle')
word2YearVec = defaultdict(list)            # { word -> LIST(year_i -> REAL) }
totalYearCounts = defaultdict(int)          # { year -> int_tot_wrd_count }
wordsInYear = defaultdict(set)            # { word -> SET(year0, year1, year2) }

#make word vectors
for year, documents in enumerate(year_book):
    for doc in documents:
        for word in doc:
            word2YearVec[word][year] += 1
            totalYearCounts[year] += 1
            wordsInyear[word].add(year)

totalYears = len(year_book)

#scale word vectors, divide each word in year, by total occurance count of that word.
for word, wordVec in word2YearVec.items():
    yearsWithWord = len(wordsInYear[word])

    for year in wordVec.keys():
        word2YearVec[word][year] = (wordVec[year] / totalYearCounts[year]) \
                                    * (totalYears / yearsWithWord)


pickle.dump(word2YearVec, open('../../data_sets/w2yv_sample.pickle', 'wb'))
