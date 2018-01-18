# -*- coding: utf-8 -*-
from gensim.corpora import WikiCorpus
import jieba
from langconv import *

def my_function():
    space = ' '
    i = 0
    l = []
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    f = open('./data/reduce_zhiwiki.txt', 'w')
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        for temp_sentence in text:
            temp_sentence = Converter('zh-hans').convert(temp_sentence)
            seg_list = list(jieba.cut(temp_sentence))
            for temp_term in seg_list:
                l.append(temp_term)
        f.write(space.join(l) + '\n')
        l = []
        i = i + 1

        if (i %200 == 0):
            print('Saved ' + str(i) + ' articles')
    f.close()

if __name__ == '__main__':
    my_function()
