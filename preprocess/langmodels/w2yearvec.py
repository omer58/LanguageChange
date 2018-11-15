from collections import defaultdict
import pickle
from math import log

LEN_YEAR_VECTOR = 1019
year_book = pickle.load(open('../../data_sets/year_book_year_dict.pickle', 'rb'))
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

print('denoising...')
WINDOW_SIZE = 5;
half_window = int( WINDOW_SIZE/2)
for word in word2YearVec:
    for i in range(len(word2YearVec[word])):
        if i - half_window > -1 and i+half_window < len(word2YearVec[word]):
            if sum(word2YearVec[word][i - half_window : i + half_window +1]) == word2YearVec[word][i]:
                word2YearVec[word][i] = 0

        elif i - WINDOW_SIZE > -1:
            if sum(word2YearVec[word][i-WINDOW_SIZE:i]) == word2YearVec[word][i]:
                word2YearVec[word][i] = 0

        elif i+ WINDOW_SIZE <= len(word2YearVec[word]):
            if sum(word2YearVec[word][i:i+WINDOW_SIZE]) == word2YearVec[word][i]:
                word2YearVec[word][i] = 0


#scale word vectors, divide each word in year, by total occurance count of that word.
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




print('pickling...')
pickle.dump(word2YearVec, open('../../data_sets/w2yv.pickle', 'wb'))
