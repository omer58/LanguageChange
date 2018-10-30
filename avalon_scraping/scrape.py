import requests
from bs4 import BeautifulSoup

page = requests.get('http://avalon.law.yale.edu/subject_menus/17th.asp')

soup = BeautifulSoup(page.text, 'html.parser')
links = soup.find_all('a')

for link in links:
    print(link.get('href'))

    page = requests.get('http://avalon.law.yale.edu/subject_menus/' + link.get('href'))

    content = BeautifulSoup(page.text, 'html.parser')
    if content.p:
        print(content.p)
