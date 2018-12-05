from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import TreebankWordTokenizer

class Cleaner:
    def __init__(self):

        self.stopwords   = set(stopwords.words('english')).union(['\''])
        self.stopwords.update({'–'})

    def _assess_indiv_(self, x):
        if len(x)>50:
            return False
        if x in self.stopwords:
            return False
        x = x.replace(u'\xa0', ' ')
        return str.lower(x.strip(punctuation))

    def clean(self, sentence):
        sent = [self._assess_indiv_(xi) for xi in sentence.split(' ')]
        return [x for x in sent if x]
