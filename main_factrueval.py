import sys
from network import Network
from dataset import Dataset
import json
import os
import layers
import sys

from load_datasets import read_dataset, make_xy, write_to_file, load_from_file
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

devset_dir = "factRuEval-2016/devset"
testset_dir = "factRuEval-2016/testset"
result_dir = "factRuEval-2016/results"

def main():
		dataset_dict = dict()
		xy_list_test = load_from_file('testudpipe.txt')
		dataset_dict['test'] = xy_list_test
			
		for i in range(3):
			xy_list_train = load_from_file('trainudpipe.txt')
			xy_list_train.extend(load_from_file('validudpipe.txt'))
			xy_list_train, xy_list_valid = train_test_split(xy_list_train, train_size = 0.5, random_state=42+i)
			dataset_dict['train'] = xy_list_train
			dataset_dict['valid'] = xy_list_valid
			write_to_file('train', dataset_dict, i)
			write_to_file('valid', dataset_dict, i)
		with open(sys.argv[1]) as f:
				params = json.load(f)

		# dataset_dict['train']=load_from_file(params['train_file'])
		# dataset_dict['valid']=load_from_file(params['valid_file'])
		for key in dataset_dict:
				print('Number of samples (sentences) in {:<5}: {}'.format(key, len(dataset_dict[key])))

		path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
		pretrained_model_path=os.path.join(path,'model/ner-model.ckpt')
		embeddings_filepath = 'rus_emb/news_upos_cbow_600_2_2018.vec.gz'

		corp = Dataset(dataset_dict, embeddings_filepath=embeddings_filepath)

		model_params = params['model_params']
		#model_params = {"filter_width": 3,"n_filters": [100, 100,], "token_embeddings_dim": 600, "char_embeddings_dim": 50, "use_crf": True, 
		#                               "embeddings_dropout": True, "pretrained_model_path": None}
		net = Network(corp, **model_params)

		#learning_params = {'epochs': 100, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'batch_size': 3, 
		#                                       'momentum': 0.9, 'max_grad': 1.0}
		learning_params = params['learning_params']
		results = net.fit(**learning_params)

if __name__=='__main__':
		main()
