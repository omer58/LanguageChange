import pickle
import glob
import json
import pprint
import random
import re
from TokenCleaner import Cleaner
from collections import defaultdict

#don't average add the words to every year
#data structure year word count


class NoDateFoundException(Exception):
    pass

#gets a string and returns the average of the years that appear in that string


def weighted_median(ques_vec):
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



def duplicate_punctuation(paragraph):
    puncs = ['–', '-', '/', ',', '.']
    for punc in puncs:
        paragraph = paragraph.replace(punc, punc+punc)

    return paragraph

def get_date(paragraph):
    paragraph = paragraph.lower()
    date = 0

    paragraph = duplicate_punctuation(paragraph) #this is done for regex serach

    #instead of not digit change it. -----------------------------------------------------
    dates = []
    dates+=re.findall(r'\D([0-9]{4})\D',paragraph)
    dates+=re.findall(r'\D([0-9]{4})$',paragraph)
    dates+=re.findall(r'$([0-9]{4})\D',paragraph)

     #r'$([0-9]{4})\D'
    if dates:
        #dates = dates.groups()
        dates = [int(date) for date in dates if date and int(date) > 1000 and int(date) < 2019]
    else:
        dates = []

    centuries = []
    centuries += re.findall(r'(1[1-9]th.century)', paragraph)
    centuries += re.findall(r'(20th.century)', paragraph)
    centuries += re.findall(r'(21st.century)', paragraph)

    if centuries:
        for date in centuries:
                century = int(date[:2]) * 100 - 50
                if int(date[:2]) == 21:
                    century = 2009
                dates.append(century)


    if dates:
        date = round(sum(dates)/len(dates))
    #date = max(set(dates), key=dates.count)




    return dates

#gets a list of paragraphs(wiki particle) and returns the date for the first
#non title paragraph.
#throws NoDateFoundException if there is no date in the document
def get_init_date(paragraphs):
    if get_date(paragraphs[0]):
        return get_date(paragraphs[0])
    elif get_date(paragraphs[1]):
        return get_date(paragraphs[1])
    else:
        res = get_date(''.join(paragraphs))

        if res == 0:
            raise NoDateFoundException()

        return res

wiki = json.loads(open('../../data_sets/wiki_lookup.json').read())

wiki2year_list = defaultdict(list)

#pprint.pprint(wiki['Wham-O'])
topics = list(wiki.keys())
#random.shuffle(topics)
i = 0;
cleaner = Cleaner()

not_found_num = 0
for topic in topics:
    i += 1
    if i%100 == 0:
        print('\r'+str(i), end='')
    article = wiki[topic]['text']

    dates = get_date(article)

    if not dates:
        #print("no dates found for: ", topic, article)
        not_found_num += 1
        continue

    dates.sort()
    wiki2year_list[topic] = dates


wiki2year = defaultdict(int)

for i in wiki2year_list:
    year = weighted_median(wiki2year_list[i])

    wiki2year[i] = wiki2year_list[i][year]

    print(i, wiki2year_list[i], wiki2year[i])
    print()


print(wiki2year)
print(not_found_num/len(topics))
print(list(wiki2year.values())[:10])



pickle.dump(wiki2year, open('../../data_sets/wiki_article_to_year.pickle', 'wb'))
