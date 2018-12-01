from scikit import spatial

def similarity(model1, model2, word):
    g1 = model1.wv[word]
    g2 = model2.wv[word]
    return 1 - spatial.distance.cosine(g0,g1)

def topn(model, word, n):
    return Ms[1].most_similar(positive='pennsylvania', topn=5)

