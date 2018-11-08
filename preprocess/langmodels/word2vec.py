from collections import defaultdict

word2YearVec = defaultdict(defaultdict)
totalWordCounts = defaultdict(int)

#make word vectors
for doc, year in input_data:
    for word in doc:
        word2YearVec[word][year] += 1
        totalWordCounts[word += 1



#scale word vectors, divide each word in year, by total occurance count of that word. 
for word, totalCount in totalWordCounts.items():
    for year, yearCount in word2YearVec[word].items():
        word2YearVec[word][year] = yearCount / totalCount
    


