from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import TreebankWordTokenizer

class Cleaner:
    def __init__():
        self.stopwords   = set(stopwords.words('english')).union(['\''])

    def _assess_indiv_(self, x):
        if len(x)>50:
            return False
        if x in self.stopwords:
            return False      
        return str.lower(x.strip(punctuation))

    def clean(self, sentence):
        sent = [self.asses_indiv(xi) for xi in sentence]
        return [x for x in sent if x]

