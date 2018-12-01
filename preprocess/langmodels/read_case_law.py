import json
import gensim
import sys
import time
import pickle
start = time.time()


data = open('/Users/omerakgul/Downloads/Pennsylvania-20180904-text/data/data.jsonl', 'r')

bucket18 = []
bucket19 = []

for idx, i in enumerate(data):

    if idx % 1000 == 0:
        sys.stdout.write("pre progress %d%%   \r" % (idx/213000*100) )
        sys.stdout.flush()



    case = json.loads(i)
    year = int(case['decision_date'][:4])
    text = []
    for op in case['casebody']['data']['opinions']:
        for sentence in op['text'].split('. '):
            text.append( gensim.utils.simple_preprocess(sentence))





    if year > 1916:
        bucket19 += text
    elif year > 1816:
        bucket18 += text


def printer(ink):
    len_ink = len(ink)
    for i, el in enumerate(ink):
        if i % 1000 == 0:
            sys.stdout.write("train progress %d%%   \r" % (idx/len_ink*100) )
            sys.stdout.flush()

        yield el

print('pickling')
with open("train_data_19.pickle", "wb") as f:
    pickle.dump(bucket19, f)

with open("train_data_19.pickle", "wb") as f:
    pickle.dump(bucket19, f)
print('pickling done.')


print("time: ", time.time() - start)
