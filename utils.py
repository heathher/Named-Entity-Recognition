import os
from gensim.parsing import PorterStemmer
from gensim.models import Word2Vec

global_stemmer = PorterStemmer()
class StemmingHelper(object):
    word_lookup = {}
    @classmethod
    def stem(cls, word):
        stemmed = global_stemmer.stem(word)
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (cls.word_lookup[stemmed].get(word, 0) + 1)
        return stemmed

    @classmethod
    def original_form(cls, word):
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(), key=lambda x: cls.word_lookup[word][x])
        else:
            return word

def makeWord2Vec(train_texts, test_texts):
    min_count = 2
    size = 50
    window = 4
    sentences = []
    for text in train_texts:
        sentences.extend(text.text.split('.'))
    model = Word2Vec(sentences, min_count=min_count, size=size, window=window)
    return model
