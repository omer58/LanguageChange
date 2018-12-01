import json
import gensim


data = open('/home/jsv/Downloads/Pennsylvania-20180904-text/data/data.jsonl', 'r')

bucket18 = []
bucket19 = []

for i in data:
    case = json.loads(i)
    year = int(case['decision_date'][:4])
    text = case['casebody'].split('. ')
    if year > 1916:
        bucket19 += text
    elif year > 1816:
        bucket18 += text
