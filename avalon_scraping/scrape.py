import requests
from bs4 import BeautifulSoup


def not_in_filter(link):
    filter = ['http', 'default', '.pdf', '21st', '#', '20th_century']
    for word in filter:
        if link.find(word) > -1:
            return False

    return True


def scrape(stack, visited):

    while stack:
        next = stack.pop()
        page = requests.get(next)

        content = BeautifulSoup(page.text, 'html.parser')

        if content.p:
            print(next)
            #print(content.p)
        else:
            links = content.find_all('a')
            for link in links:
                link = link.get('href')
                if link and not_in_filter(link):

                    link = 'http://avalon.law.yale.edu/subject_menus/' + link
                    if link.find('..') > -1 and  link not in visited and link not in stack:
                        visited.add(link)
                        stack.append(link)
                        print('append ', link)





if __name__ == '__main__':
    visited = set()
    stack = ['http://avalon.law.yale.edu/subject_menus/15th.asp']
    scrape(stack, visited)
