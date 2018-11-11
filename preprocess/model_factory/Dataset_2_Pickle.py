import pickle
import glob
import json
import pprint
import random
import re

'''
julians outdated stuff

#packages are defined as [modelinfo_1][modelinfo_2][label]
#here that is [word2vec][chinese restaurant][label]
#add more models as needed


train_legal_pkg
train_books_pkg
train_newsp_pkg
test_legal_pkg
test_books_pkg
test_newsp_pkg

if not pkgs exist in data_sets folder:

    for file in glob('../data_sets/*'):
        create models for each pkg. Which models do we want?
            chinese restaurant
            pure words
        pkg[-1]=label #ie timeslice in discrete time space
        add to appropriatePkg

    pickle all built pkgs_unfinished

else:
    unpickle packages



if have unfinished packages:
    train word2vec
    repickle

'''
class NoDateFoundException(Exception):
    pass

#gets a string and returns the average of the years that appear in that string




def duplicate_punctuation(paragraph):
    puncs = ['–', '-', '/', ',', '.']
    for punc in puncs:
        paragraph = paragraph.replace(punc, punc+punc)

    return paragraph

def get_date(paragraph):
    paragraph = paragraph.lower()
    date = 0

    paragraph = duplicate_punctuation(paragraph) #this is done for regex serach

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



    return date

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

wiki = json.loads(open('../../data_sets/wiki_lookup_sample.json').read())
year_book = [] #dictionary of years and the document that belongs to that year

for i in range(1019):
    year_book.append([])


#pprint.pprint(wiki['Wham-O'])
topics = list(wiki.keys())
#random.shuffle(topics)
i = 0;
for topic in topics:
    i += 1
    if i%100 == 0:
        print('\r'+str(i), end='')
    paragraphs = wiki[topic]['text'].split('\n')
    try:
        prev_date = get_init_date(paragraphs)
    except NoDateFoundException:
        continue

    year_book[prev_date-1000].append(paragraphs[1])




    for paragraph in paragraphs[2:]:
        if len(paragraph) > 50:
            date = get_date(paragraph)

            if not date:
                date = prev_date

            #print('\n\n')
            #print('p: ', paragraph)
            #print('date: ', date)
            #print('\n\n')

            prev_date = date

            #add to data structure
            year_book[date-1000].append(paragraph)

            if date == 1000:
                print(date)





pickle.dump(year_book, open('../../data_sets/year_book_sample.pickle', 'wb'))
