import sys
from network import Network
from dataset import Dataset
import json
import os
import layers

from load_datasets import read_dataset, make_xy, write_to_file, load_from_file

devset_dir = "factRuEval-2016/devset"
testset_dir = "factRuEval-2016/testset"
result_dir = "factRuEval-2016/results"

def main():
	dataset_dict = dict()
	#train_tokens = list(read_dataset(devset_dir, 'tokens'))
	#test_tokens = list(read_dataset(testset_dir, 'tokens'))
	#train_objects = list(read_dataset(devset_dir, 'objects'))
	#test_objects = list(read_dataset(testset_dir, 'objects'))

	#xy_list_train = make_xy(train_tokens, train_objects)
	#print(xy_list_train[:2])
	#xy_list_test = make_xy(test_tokens, test_objects)
	
	xy_list_train = load_from_file('train.txt')
	xy_list_valid = load_from_file('valid.txt')
	xy_list_test = load_from_file('test.txt')
	dataset_dict['train'] = xy_list_train
	dataset_dict['valid'] = xy_list_valid
	dataset_dict['test'] = xy_list_test

	#write_to_file('train', dataset_dict)
	#write_to_file('valid', dataset_dict)
	#write_to_file('test', dataset_dict)
	
	for key in dataset_dict:
		print('Number of samples (sentences) in {:<5}: {}'.format(key, len(dataset_dict[key])))

	corp = Dataset(dataset_dict, embeddings_filepath='rus_emb/news_upos_cbow_600_2_2018.vec.gz')


	model_params = {"filter_width": 6,"n_filters": [100, 100,], "token_embeddings_dim": 600, "char_embeddings_dim": 25, "use_crf": True, "embeddings_dropout": True}
	net = Network(corp, **model_params)

	learning_params = {'epochs': 150, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'batch_size': 10, 
						'momentum': 0.9, 'max_grad': 5.0}

	results = net.fit(**learning_params)

if __name__=='__main__':
	main()
