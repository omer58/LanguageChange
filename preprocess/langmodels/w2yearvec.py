from collections import defaultdict
import pickle

year_book = pickle.loads('../../data_sets/year_book.pickle')
word2YearVec = defaultdict(defaultdict)
totalYearCounts = defaultdict(int)
wordsInYear = defaultdict(set())

#make word vectors
for i, documents in enumerate(year_book):
    year = i+1000
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


pickle.dump(word2YearVec, open('../../data_sets/w2yv.p', 'wb'))

