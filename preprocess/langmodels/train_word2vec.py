import json
import gensim
import sys
import time
import pickle
<<<<<<< HEAD
import align
import numpy as np
=======
start = time.time()
>>>>>>> parent of 906c7ba... ALIGNS_VECTORS

with open("train_data_19.pickle", "rb") as f:
    bucket19 = pickle.load(f)
#with open("train_data_18.pickle", 'rb') as f:
#    bucket18 = pickle.load(f)
print('model 19 start')

<<<<<<< HEAD
def w2v_make(num, last=False):
    start = time.time()
    vocab = set()
    num = str(num)
    #with open("train_data_"+num+".pickle", "rb") as f:
    #    bucket = pickle.load(f)

    bucket = np.load('train_data_'+ num +'_np').tolist()
=======
model = gensim.models.Word2Vec(bucket19, size=150, window=10, min_count=2, workers=10)
prev = 0
tot_len = len(bucket19);
for i in range(5000, tot_len, 5000):
    model.train(bucket19[prev:i],total_examples=5000,epochs=10)
    sys.stdout.write("train progress %d%%   \r" % (i/tot_len*100) )
    sys.stdout.flush()
    prev = i
>>>>>>> parent of 906c7ba... ALIGNS_VECTORS

model.train(bucket19[prev:],total_examples=len(bucket19[prev:]),epochs=10)


model.save('w2v_19.model')

print('bucket 19 done.')

#model = gensim.models.Word2Vec (bucket18, size=150, window=10, min_count=2, workers=10)
#model.train(bucket18,total_examples=len(bucket18),epochs=10)
#model.save('w2v_18.model')

print("time: ", time.time() - start)
