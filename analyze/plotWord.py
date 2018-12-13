import pickle as P
import numpy as np
from matplotlib import pyplot as plt

D=P.load(open("../data_sets/w2yv_dict_wordnormalized.pickle", 'rb'))
V=np.load(open("../data_sets/w2yv_vals_wordnormalized.npy",'rb'))

def picture(W):
    plt.clf()
    for w in W:
        if w in D[w]:
            plt.plot(V[D[w]], label=w)
    plt.legend()
    plt.show()


