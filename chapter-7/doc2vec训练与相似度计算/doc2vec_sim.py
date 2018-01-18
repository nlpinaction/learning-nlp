import gensim.models as g
import codecs
import numpy
import numpy as np

model_path = './data/zhiwiki_news.doc2vec'
start_alpha = 0.01
infer_epoch = 1000
docvec_size = 192


def simlarityCalu(vector1, vector2):
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity


def doc2vec(file_name, model):
    import jieba
    doc = [w for x in codecs.open(file_name, 'r', 'utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    return doc_vec_all


if __name__ == '__main__':
    model = g.Doc2Vec.load(model_path)
    p1 = './data/P1.txt'
    p2 = './data/P2.txt'
    P1_doc2vec = doc2vec(p1, model)
    P2_doc2vec = doc2vec(p2, model)
    print(simlarityCalu(P1_doc2vec, P2_doc2vec))
