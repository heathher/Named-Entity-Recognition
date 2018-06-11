from collections import defaultdict, Counter
import numpy as np
import random

SPECIAL_TOKENS = ['<PAD>', '<UNK>']
SPECIAL_TAGS = ['<PAD>']
SEED = 32

np.random.seed(SEED)
random.seed(SEED)


class Dataset:
	def __init__(self, dataset):
		if dataset is not None:
			self.dataset = dataset
			self.token_dict = Vocabulary(self.get_tokens())
			self.tag_dict = Vocabulary(self.get_tags(), is_tags=True)
			self.char_dict = Vocabulary(self.get_characters())
			self.embeddings = None
		else:
			print("Load dataset")
	
	def get_tokens(self, data_type='train'):
		for tokens, _ in self.dataset[data_type]:
			for token in tokens:
				yield token

	def get_tags(self, data_type=None):
		if data_type is None:
			data_types = self.dataset.keys()
		else:
			data_types = [data_type]
		for data_type in data_types:
			for _, tags in self.dataset[data_type]:
				for tag in tags:
					yield tag

	def get_characters(self, data_type='train'):
		for tokens, _ in self.dataset[data_type]:
			for token in tokens:
				for character in token:
					yield character

	def batch_generator(self, batch_size, dataset_type='train', shuffle=True,
						allow_smaller_last_batch=True):
		tokens_tags_pairs = self.dataset[dataset_type]
		n_samples = len(tokens_tags_pairs)
		if shuffle:
			order = np.random.permutation(n_samples)
		else:
			order = np.arange(n_samples)
		n_batches = n_samples // batch_size
		if allow_smaller_last_batch and n_samples % batch_size:
			n_batches += 1
		for k in range(n_batches):
			batch_start = k * batch_size
			batch_end = min((k + 1) * batch_size, n_samples)
			x_batch = [tokens_tags_pairs[ind][0] for ind in order[batch_start: batch_end]]
			y_batch = [tokens_tags_pairs[ind][1] for ind in order[batch_start: batch_end]]
			x, y = self.tokens_batch_to_numpy_batch(x_batch, y_batch)
			yield x, y

	def tokens_batch_to_numpy_batch(self, batch_x, batch_y=None):
		x = dict()
		# Determine dimensions
		batch_size = len(batch_x)
		max_utt_len = max([len(utt) for utt in batch_x])

		max_utt_len = max(max_utt_len, 2)
		max_token_len = max([len(token) for utt in batch_x for token in utt])

		x['token'] = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.token_dict['<PAD>']
		x['char'] = np.ones([batch_size, max_utt_len, max_token_len], dtype=np.int32) * self.char_dict['<PAD>']

		# Capitalization
		x['capitalization'] = np.zeros([batch_size, max_utt_len], dtype=np.float32)
		for n, utt in enumerate(batch_x):
			for k, tok in enumerate(utt):
				if len(tok) > 0 and tok[0].isupper():
					x['capitalization'][n, k] = 1

		# Prepare x batch
		for n, utterance in enumerate(batch_x):
			x['token'][n, :len(utterance)] = self.token_dict.toks2idxs(utterance)
			for k, token in enumerate(utterance):
				x['char'][n, k, :len(token)] = self.char_dict.toks2idxs(token)

		# Mask for paddings
		x['mask'] = np.zeros([batch_size, max_utt_len], dtype=np.float32)
		for n in range(batch_size):
			x['mask'][n, :len(batch_x[n])] = 1

		# Prepare y batch
		if batch_y is not None:
			y = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.tag_dict['<PAD>']
		else:
			y = None

		if batch_y is not None:
			for n, tags in enumerate(batch_y):
				y[n, :len(tags)] = self.tag_dict.toks2idxs(tags)

		return x, y
			 

class Vocabulary:
	def __init__(self, tokens=None, is_tags=False):
		if is_tags:
			special_tokens = SPECIAL_TAGS
			self._t2i = dict()
		else:
			special_tokens = SPECIAL_TOKENS
			default_ind = special_tokens.index('<UNK>')
			self._t2i = defaultdict(lambda: default_ind)
		self._i2t = list()
		self.frequencies = Counter()
		self.counter = 0
		for token in special_tokens:
			self._t2i[token] = self.counter
			self.frequencies[token] += 0
			self._i2t.append(token)
			self.counter += 1
		if tokens is not None:
			self.update_dict(tokens)

	def update_dict(self, tokens):
		for token in tokens:
			if token not in self._t2i:
				self._t2i[token] = self.counter
				self._i2t.append(token)
				self.counter += 1
			self.frequencies[token] += 1

	def tok2idx(self, tok):
		return self._t2i[tok]

	def toks2idxs(self, toks):
		return [self._t2i[tok] for tok in toks]

	def batch_idxs2batch_toks(self, b_idxs, filter_paddings=False):
		return [self.idxs2toks(idxs, filter_paddings) for idxs in b_idxs]

	def idx2tok(self, idx):
		return self._i2t[idx]

	def idxs2toks(self, idxs, filter_paddings=False):
		toks = []
		for idx in idxs:
			if not filter_paddings or idx != self.tok2idx('<PAD>'):
				toks.append(self._i2t[idx])
		return toks

	def __len__(self):
		return self.counter

	def __getitem__(self, key):
		return self._t2i[key]
