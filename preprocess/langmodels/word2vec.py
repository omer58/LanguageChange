from collections import defaultdict

word2YearVec = defaultdict(defaultdict)
totalYearCounts = defaultdict(int)

#make word vectors
for doc, year in input_data:
    for word in doc:
        word2YearVec[word][year] += 1
        totalYearCounts[year] += 1

totalYears = len(totalYearCounts.keys())

#scale word vectors, divide each word in year, by total occurance count of that word. 
for word, wordVec in word2YearVec.items():
    yearsWithWord = len(wordVec)

    for year in wordVec.keys():
        word2YearVec[word][year] = (wordVec[year] / totalYearCounts[year]) \
                                    * (totalYears / yearsWithWord)

return shit

