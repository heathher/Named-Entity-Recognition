import sys
from network import Network
from dataset import Dataset
import json
import os
import layers

def main():
	data_types = ['train', 'test', 'valid']
	dataset_dict = dict()
	for data_type in data_types:
		with open('conll2003/' + data_type + '.txt') as f:
			xy_list = list()
			tokens = list()
			tags = list()
			for line in f:
				items = line.split()
				if len(items) > 1 and '-DOCSTART-' not in items[0]:
					token, tag = items
					if token[0].isdigit():
						tokens.append('#')
					else:
						tokens.append(token)
					tags.append(tag)
				elif len(tokens) > 0:
					xy_list.append((tokens, tags,))
					tokens = list()
					tags = list()
			dataset_dict[data_type] = xy_list
	for key in dataset_dict:
		print('Number of samples (sentences) in {:<5}: {}'.format(key, len(dataset_dict[key])))

	corp = Dataset(dataset_dict)

	model_params = {"filter_width": 3,"n_filters": [200, 200,], "token_embeddings_dim": 100, "char_embeddings_dim": 30, "use_crf": True, "embeddings_dropout": True}
	net = Network(corp, **model_params)

	learning_params = {'epochs': 100, 'dropout_rate': 0.5, 'learning_rate': 0.015, 'batch_size': 10, 'learning_rate_decay': 0.05}
	results = net.fit(**learning_params)

if __name__=='__main__':
	main()
