from __future__ import unicode_literals
from collections import namedtuple
import os
import re

DEVSET_DIR = "/devset"
TESTSET_DIT = "/testset"
RESULT_DIR = "/results"

FactRu = namedtuple('FactRu', ['id', 'text'])

def load_data(path):
    with open(path) as file:
        return file.read()

def read_dataset(dir_path, filetype):
    for filename in os.listdir(dir_path):
        match = re.match('book_(\d+)\.'+filetype, filename)
        if match:
            id = int(match.group(1))
            path = os.path.join(dir_path, filename)
            text = load_data(path)
            yield FactRu(id, text)

#only for PERSON
def make_labels(objects):
    result = []
    tokenized_objects = tokenize(objects)
    for _ in tokenized_objects:
        if _.text[1] == 'Person':
            index = _.text.index('#')
            result.append(FactRu(_.id, ['PER', _.text[index+1:]]))
        elif _.text[1] == 'Location':
            index = _.text.index('#')
            result.append(FactRu(_.id, ['LOC', _.text[index+1:]]))
        elif _.text[1] == 'Org':
            index = _.text.index('#')
            result.append(FactRu(_.id, ['ORG', _.text[index+1:]]))
    return result

def tokenizer(text):
    _split = re.compile(r'([^\w_-]|[+])', re.UNICODE).split
    return [t for t in _split(text) if t and not t.isspace()]

def tokenize(texts):
    result = []
    for text in texts:
        for line in text.text.splitlines():
            result.append(FactRu(text.id, tokenizer(line)))
    return result

def load_dataset(dir):
    train_text = list(read_dataset(dir+DEVSET_DIR, 'txt'))
    test_text = list(read_dataset(dir+TESTSET_DIT, 'txt'))
    train_objects = list(read_dataset(dir+DEVSET_DIR, 'objects'))
    test_objects = list(read_dataset(dir+TESTSET_DIT, 'objects'))
    train_labels = make_labels(train_objects)
    test_labels = make_labels(test_objects)
    return [[train_text, train_labels], [test_text, test_labels]]