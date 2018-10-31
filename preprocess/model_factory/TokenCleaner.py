from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import TreebankWordTokenizer

class Cleaner:
    def __init__():
        self.stopwords   = set(stopwords.words('english')).union(['\''])
        self.digits      = set(str(i) for i in range(10)))
        self.t           = TreebankWordTokenizer().tokenize

    def tokenize(self, x):
        return self.t(x)

    def assess_indiv(self, x):
        if len(x)>50:
            return False
        if x in self.stopwords:
            return False
        for xi in x:
            if xi in self.digits:
                return False
        return str.lower(x.strip(punctuation))

    def clean(self, x):
        return [self.asses_indiv(xi) for xi in x]

    def tokenize_clean(self, sentence):
        return self.assess(self.t(x))
