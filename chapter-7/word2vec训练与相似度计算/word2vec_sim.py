# -*- coding: utf-8 -*-
import codecs
import numpy
import gensim
import numpy as np
from keyword_extract import *

wordvec_size=192
def get_char_pos(string,char):
    chPos=[]
    try:
        chPos=list(((pos) for pos,val in enumerate(string) if(val == char)))
    except:
        pass
    return chPos

def word2vec(file_name,model):
    with codecs.open(file_name, 'r') as f:
        word_vec_all = numpy.zeros(wordvec_size)
        for data in f:
            space_pos = get_char_pos(data, ' ')
            first_word=data[0:space_pos[0]]
            if model.__contains__(first_word):
                word_vec_all= word_vec_all+model[first_word]

            for i in range(len(space_pos) - 1):
                word = data[space_pos[i]:space_pos[i + 1]]
                if model.__contains__(word):
                    word_vec_all = word_vec_all+model[word]
        return word_vec_all

def simlarityCalu(vector1,vector2):
    vector1Mod=np.sqrt(vector1.dot(vector1))
    vector2Mod=np.sqrt(vector2.dot(vector2))
    if vector2Mod!=0 and vector1Mod!=0:
        simlarity=(vector1.dot(vector2))/(vector1Mod*vector2Mod)
    else:
        simlarity=0
    return simlarity

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('data/zhiwiki_news.word2vec')
    p1 = './data/P1.txt'
    p2 = './data/P2.txt'
    p1_keywords = './data/P1_keywords.txt'
    p2_keywords = './data/P2_keywords.txt'
    getKeywords(p1, p1_keywords)
    getKeywords(p2, p2_keywords)
    p1_vec=word2vec(p1_keywords,model)
    p2_vec=word2vec(p2_keywords,model)

    print(simlarityCalu(p1_vec,p2_vec))
