import json
import gensim
import sys
import time
import pickle
import numpy as np


def run_test(file_path='/Users/omerakgul/Downloads/Pennsylvania-20180904-text/data/data.jsonl', num_cases=5000):
    start = time.time()
    data = open(file_path, 'r')
    bucket18 = []
    bucket19 = []
    count18 = 0
    count19 = 0

    for idx, i in enumerate(data):

        if idx % 1000 == 0:
            sys.stdout.write("pre progress %d%%   \r" % (idx/213000*100) )
            sys.stdout.flush()

        if count18>num_cases and count19>num_cases:
            break

        case = json.loads(i)
        year = int(case['decision_date'][:4])

        if year > 1916 and count19 > num_cases:
            continue
        if year > 1816 and count18 > num_cases:
            continue
        if year < 1816:
            continue

        text = []
        for op in case['casebody']['data']['opinions']:
            for sentence in op['text'].split('. '):
                text.append( gensim.utils.simple_preprocess(sentence))

        if year > 1916:
            bucket19 += text
            count19 +=1
        elif year > 1816:
            bucket18 += text
            count18 +=1


    def printer(ink):
        len_ink = len(ink)
        for i, el in enumerate(ink):
            if i % 1000 == 0:
                sys.stdout.write("train progress %d%%   \r" % (idx/len_ink*100) )
                sys.stdout.flush()

            yield el

    print('saving')
    np.save('train_data_19_np', np.array(bucket19))
    np.save('train_data_18_np', np.array(bucket18))
    print('saved')
    print("time: ", time.time() - start)







def run(file_path='/Users/omerakgul/Downloads/Pennsylvania-20180904-text/data/data.jsonl'):
    start = time.time()
    data = open(file_path, 'r')
    bucket17 = []
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
                text.append( ' '.join(gensim.utils.simple_preprocess(sentence)) + '\n' )

        if   year > 1918:
            bucket19 += text
        elif year > 1818:
            bucket18 += text
        elif year > 1718:
            bucket17 += text


    def printer(ink):
        len_ink = len(ink)
        for i, el in enumerate(ink):
            if i % 1000 == 0:
                sys.stdout.write("train progress %d%%   \r" % (idx/len_ink*100) )
                sys.stdout.flush()

            yield el
    data.close()
    #np.save('train_data_18_np', np.array(bucket18))

    print('saving')
    if bucket17:
        with open('train_data_17_np.txt','w') as F:
            for i in bucket17:
                F.write(i)
        print('bucket 17 done')
    if bucket18:
         with open('train_data_18_np.txt','w') as F:
             for i in bucket18:
                 F.write(i)
         print('bucket 18 done')
    if bucket19:
        with open('train_data_19_np.txt','w') as F:
         for i in bucket19:
             F.write(i)
        print('bucket 19 done')

    print("time: ", time.time() - start)
