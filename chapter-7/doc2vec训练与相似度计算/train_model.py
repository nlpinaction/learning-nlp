#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim.models as g
from gensim.corpora import WikiCorpus
import logging
from langconv import *

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

docvec_size=192
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        import jieba
        for content, (page_id, title) in self.wiki.get_texts():
            yield g.doc2vec.LabeledSentence(words=[w for c in content for w in jieba.cut(Converter('zh-hans').convert(c))], tags=[title])

def my_function():
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    documents = TaggedWikiDocument(wiki)

    model = g.Doc2Vec(documents, dm=0, dbow_words=1, size=docvec_size, window=8, min_count=19, iter=5, workers=8)
    model.save('data/zhiwiki_news.doc2vec')

if __name__ == '__main__':
    my_function()

