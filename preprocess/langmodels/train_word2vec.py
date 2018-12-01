import json
import gensim
import sys
import time
import pickle
import align
import numpy as np

START_CENTURY = 18
LAST_CENTURY = 19

def w2v_make(num, last=False):
    start = time.time()
    vocab = set()


    num = str(num)
 
    #with open("train_data_"+num+".pickle", "rb") as f:
    #    bucket = pickle.load(f)
    bucket = np.load('train_data_'+num+'_np.npy').tolist()
    print('len of bucket is ',len(bucket))

    print('model',num,' start')
    model = gensim.models.Word2Vec(bucket, size=300, window=5, min_count=4, workers=10, negative=10)
    prev = 0
    tot_len = len(bucket);

    for epoch in range(10):
        for i in range(5000, tot_len, 5000):
            model.train(bucket[prev:i],total_examples=5000, epochs=1)
            sys.stdout.write("epoch "+str(epoch)+", train progress %d%%   \r" % (i/tot_len*100) )
            sys.stdout.flush()
            prev = i
        model.train(bucket[prev:],total_examples=len(bucket[prev:]), epochs=1)

    model.save('w2v_'+num+'.model')
    print("time: ", time.time() - start)
    print('bucket ',num,' done.')

    if last:
        print('Compiling vocab...')
        for sentence in bucket:
            for word in sentence:
                vocab.add(word)
        print('...finished')

    return model, vocab



def alg(l_m, w):
    # centered around final embeddings words
    base = l_m[-1]

    for i in range(len(l_m) - 1):
        other = l_m[i]
        l_m[i] = align.smart_procrustes_align_gensim(base, other, words=w)

    return l_m

def run():
    print('Building and aligning models. This method saves models to w2v_aligned_x.model and returns a LIST of MODELS')
    models = []
    for b_num in range(START_CENTURY, LAST_CENTURY): #NON-INCLUSIVE
        m, _ = w2v_make(b_num)
        models.append(m)

    m, w = w2v_make(LAST_CENTURY, last=True)
    models.append(m)

    models = alg(models, w)
    print('model alignment finished')
    print('saving models')
    for ii in range(len(models)):
        print(str(ii), ' of ', str(len(models)),'...')
        models[ii].save('w2v_aligned_'+str(ii)+'.model')
    return models
