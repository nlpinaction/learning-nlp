#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs
import numpy as np

def SimlarityCalu(Vector1,Vector2):
    Vector1Mod=np.sqrt(Vector1.dot(Vector1))
    Vector2Mod=np.sqrt(Vector2.dot(Vector2))
    if Vector2Mod!=0 and Vector1Mod!=0:
        simlarity=(Vector1.dot(Vector2))/(Vector1Mod*Vector2Mod)
    else:
        simlarity=0
    return simlarity

#parameters
model='toy_data/model.bin'
test_docs='toy_data/p11.txt'
output_file='toy_data/test_vectors.txt'

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)
test_docs = [ x.strip().split() for x in codecs.open(test_docs, 'r', 'utf-8').readlines()]

#infer test vectors
output = open(output_file, 'w')
a=[]

for d in test_docs:
    output.write( ' '.join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + '\n' )
    a.append(m.infer_vector(d, alpha=start_alpha, steps=infer_epoch))
output.flush()
output.close()
print(SimlarityCalu(a[0],a[1]))