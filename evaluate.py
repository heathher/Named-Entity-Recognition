from time import time
from load_datasets import load_dataset
from utils import makeWord2Vec
import os

PREDS_FNAME = 'preds.tsv'
DIR = 'factRuEval-2016'
EMBEDDINGS = 'embeddings_model'

def main():
    datasets = load_dataset(DIR)
    train_texts, train_labels = datasets[0][0], datasets[0][1]
    test_texts, test_labels = datasets[1][0], datasets[1][1]
    model = makeWord2Vec(train_texts, test_texts)
    model.save(EMBEDDINGS)
    print(model)

if __name__=='__main__':
    main()