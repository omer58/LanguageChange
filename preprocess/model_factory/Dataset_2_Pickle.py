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


def dump_w2yv(year_book, file):
    file.write(str(len(year_book)) + '\n')

    for i in year_book:
        file.write(str(i) + '\n')

def load_w2yv(file):
    year_book = []
    for i in range(int(file.readline())):
        year_book.append(defaultdict(int))

    for i,l in enumerate(file):
        print(l)
        for ent in l[1:-2].split(','):
            if ent:
                year_book[i][ent.split("'")[1]] = int(ent.split(':')[1])





    return year_book

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


if __name__ == "__main__":

    wiki = json.loads(open('../../data_sets/wiki_lookup_sample.json').read())
    year_book = [] #dictionary of years and the document that belongs to that year

    for i in range(1019):
        year_book.append(defaultdict(int))


    #pprint.pprint(wiki['Wham-O'])
    topics = list(wiki.keys())
    #random.shuffle(topics)
    i = 0;
    cleaner = Cleaner()

    for topic in topics:
        i += 1
        if i%100 == 0:
            print('\r'+str(i), end='')
        paragraphs = wiki[topic]['text'].split('\n')
        try:
            prev_date = get_init_date(paragraphs)
        except NoDateFoundException:
            continue

        para = cleaner.clean(paragraphs[1])

        for year in prev_date:
            for word in para:
                year_book[year-1000][word] += 1
        #year_book[prev_date-1000].append(paragraphs[1])




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


                paragraph = cleaner.clean(paragraph)

                for year in date:
                    for word in paragraph:
                        year_book[year-1000][word] += 1


    dump_w2yv(year_book, open("../../data_sets/year_book_sample.txt", 'wb'))
    #pickle.dump(year_book, open('../../data_sets/year_book.pickle', 'wb'))
