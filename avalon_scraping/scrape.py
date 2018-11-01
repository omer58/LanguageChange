import requests
from bs4 import BeautifulSoup
import os
import pickle
import re


def not_in_filter(link):
    filter = ['http', 'default', '.pdf', '21st', '#']
    for word in filter:
        if link.find(word) > -1:
            return False

    return True

def check_end():
    ls = os.listdir('./')

    if 'cont' not in ls:
        return True
    return False

def save(stack, visited):
    pickle.dump(stack, open('stack.pickle', 'wb'))
    pickle.dump(visited, open('visited.pickle', 'wb'))

def find_text(tag):
    classes = tag.get('class', [])
    if tag.name in ['p', 'h1', 'h2', 'h4', 'h5', 'br']:
        return True

    if 'document-title' in classes:
        return True

    return False




def scrape(stack, visited):

    loop = 0
    while stack:

        if check_end():
            save(stack, visited)
            return

        if loop % 2000:
            save(stack, visited)


        next = stack.pop()
        page = requests.get(next)

        content = BeautifulSoup(page.text, 'html.parser')

        all_text = ''
        if content.p:
            dates = []
            for text in content.find_all(find_text):
                try:
                    text = str(text)
                except RecursionError:
                    continue

                all_text += text
                #print(text)
                new_dates = re.search(r'( [0-9]{4} )', text)
                if not new_dates:
                    new_dates = re.search(r'( [0-9]{4})', text)
                if not new_dates:
                    new_dates = re.search(r'([0-9]{4} )', text)
                if not new_dates:
                    new_dates = re.search(r'([0-9]{2}th)', text)
                if not new_dates:
                    continue


                new_dates = new_dates.groups()
                #new_dates = [int(date) for date in new_dates if int(date) > 1500 and int(date) < 2019]
                if new_dates:
                    dates += new_dates

            if dates:
                date = max(set(dates), key=dates.count)
            elif re.search(r'([0-9]{4})', next):
                date = re.search(r'([0-9]{2}th)', next).groups()[0]
            elif re.search(r'([0-9]{2}th)', next):
                date = re.search(r'([0-9]{2}th)', next).groups()[0]
            else:
                print(next)

            print(date)
            date = date.replace(' ', '')
            next = next.replace('/', '-')
            with open('all_docs/'+date+next, 'w+') as file:
                file.write(all_text)
            print('----------------------------------')
            #print(content.p)




        else:
            links = content.find_all('a')
            for link in links:
                link = link.get('href')
                if link and not_in_filter(link):

                    link = 'http://avalon.law.yale.edu/subject_menus/' + link
                    if link not in visited and link not in stack:# and link.find('..') > -1:
                        visited.add(link)
                        stack.append(link)
                        #print('append ', link)

        loop += 1





if __name__ == '__main__':
    if os.path.isfile('stack.pickle'):
        stack = pickle.load(open( 'stack.pickle', "rb" ))
    else:
        stack = ['http://avalon.law.yale.edu/subject_menus/15th.asp']

    if os.path.isfile('visited.pickle'):
        visited = pickle.load(open( 'visited.pickle', "rb" ))
    else:
        visited = set()

    scrape(stack, visited)
