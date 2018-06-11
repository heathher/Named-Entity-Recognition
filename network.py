import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import os

from layers import character_embedding_network, embedding_layer, biLSTM
from evaluation import precision_recall_f1


class Network:
	def __init__(self, corpus, n_filters=(128, 256), filter_width=3, token_embeddings_dim=128, char_embeddings_dim=50,
				use_char_embeddins=True, embeddings_dropout=False, use_crf=False, char_filter_width=5):

		tf.reset_default_graph()

		n_tags = len(corpus.tag_dict)
		n_tokens = len(corpus.token_dict)
		n_chars = len(corpus.char_dict)

		# Create placeholders
		x_word = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
		x_char = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
		y_true = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')
		mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='mask')
		learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
		dropout_ph = tf.placeholder_with_default(1.0, shape=[])
		training_ph = tf.placeholder_with_default(False, shape=[])
		learning_rate_decay_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_decay')

		# Embeddings
		with tf.variable_scope('Embeddings'):
			w_emb = embedding_layer(x_word, n_tokens=n_tokens, token_embedding_dim=token_embeddings_dim)
			if use_char_embeddins:
				c_emb = character_embedding_network(x_char, n_characters=n_chars, char_embedding_dim=char_embeddings_dim,
													filter_width=char_filter_width)
				emb = tf.concat([w_emb, c_emb], axis=-1)
			else:
				emb = w_emb
		# Dropout for embeddings
		if embeddings_dropout:
			emb = tf.layers.dropout(emb, dropout_ph, training=training_ph)
		# Make bi-LSTM
			units = biLSTM(emb, n_filters)

		# Classifier
		with tf.variable_scope('Classifier'):
			logits = tf.layers.dense(units, n_tags, kernel_initializer=xavier_initializer())
		if use_crf:
			sequence_lengths = tf.reduce_sum(mask, axis=1)
			log_likelihood, trainsition_params = tf.contrib.crf.crf_log_likelihood(logits, y_true, sequence_lengths)
			loss_tensor = -log_likelihood
			predictions = None
		else:
			ground_truth_labels = tf.one_hot(y_true, n_tags)
			loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
			loss_tensor = loss_tensor * mask
			predictions = tf.argmax(logits, axis=-1)

		loss = tf.reduce_mean(loss_tensor)

		# Initialize session
		sess = tf.Session()

		self._use_crf = use_crf
		self.summary = tf.summary.merge_all()
		self._learning_rate_decay_ph = learning_rate_decay_ph
		self._x_w = x_word
		self._x_c = x_char
		self._y_true = y_true
		self._y_pred = predictions
		self._learning_rate_ph = learning_rate_ph
		self._dropout = dropout_ph
		self._loss = loss
		self._sess = sess
		self.corpus = corpus
		self._loss_tensor = loss_tensor
		self._use_dropout = embeddings_dropout
		self._training_ph = training_ph
		if use_crf:
			self._logits = logits
			self._trainsition_params = trainsition_params
			self._sequence_lengths = sequence_lengths
		self._train_op = self.get_train_op(loss, learning_rate_ph, lr_decay_rate=learning_rate_decay_ph)
		sess.run(tf.global_variables_initializer())
		self._mask = mask

	def get_train_op(self, loss, learning_rate, learnable_scopes=None, lr_decay_rate=None):
		global_step = tf.Variable(0, trainable=False)
		try:
			n_training_samples = len(self.corpus.dataset['train'])
		except TypeError:
			n_training_samples = 1024
		batch_size = tf.shape(self._x_w)[0]
		decay_steps = tf.cast(n_training_samples / batch_size, tf.int32)
		if lr_decay_rate is not None:
			learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=decay_steps,
													   decay_rate=lr_decay_rate, staircase=True)
			self._learning_rate_decayed = learning_rate
		variables = self.get_trainable_variables(learnable_scopes)
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(extra_update_ops):
			train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=variables)
		return train_op

	@staticmethod
	def get_trainable_variables(trainable_scope_names=None):
		vars = tf.trainable_variables()
		if trainable_scope_names is not None:
			vars_to_train = list()
			for scope_name in trainable_scope_names:
				for var in vars:
					if var.name.startswith(scope_name):
						vars_to_train.append(var)
			return vars_to_train
		else:
			return vars

	def fit(self, batch_gen=None, batch_size=32, learning_rate=1e-3, epochs=1, dropout_rate=0.5, learning_rate_decay=1):
		for epoch in range(epochs):
			print('Epoch {}'.format(epoch))
			if batch_gen is None:
				batch_generator = self.corpus.batch_generator(batch_size, dataset_type='train')
			for x, y in batch_generator:
				feed_dict = self.fill_feed_dict(x,y,learning_rate,dropout_rate=dropout_rate,training=True,
												 learning_rate_decay=learning_rate_decay)
				self._sess.run(self._train_op, feed_dict=feed_dict)
			self.eval_conll('valid', print_results=True)
		self.eval_conll(dataset_type='train', short_report=False)
		self.eval_conll(dataset_type='valid', short_report=False)
		results = self.eval_conll(dataset_type='test', short_report=False)
		return results

	def eval_conll(self, dataset_type='test', print_results=True, short_report=True):
		y_true_list = list()
		y_pred_list = list()
		print('Eval on {}:'.format(dataset_type))
		for x, y_gt in self.corpus.batch_generator(batch_size=32, dataset_type=dataset_type):
			y_pred = self.predict(x)
			y_gt = self.corpus.tag_dict.batch_idxs2batch_toks(y_gt, filter_paddings=True)
			for tags_pred, tags_gt in zip(y_pred, y_gt):
				for tag_predicted, tag_ground_truth in zip(tags_pred, tags_gt):
					y_true_list.append(tag_ground_truth)
					y_pred_list.append(tag_predicted)
				y_true_list.append('O')
				y_pred_list.append('O')
		return precision_recall_f1(y_true_list, y_pred_list, print_results, short_report)

	def fill_feed_dict(self,x,y_t=None,learning_rate=None,training=False,dropout_rate=1.0,learning_rate_decay=1.0):
		feed_dict = dict()
		feed_dict[self._x_w] = x['token']
		feed_dict[self._x_c] = x['char']
		feed_dict[self._mask] = x['mask']
		feed_dict[self._training_ph] = training
		if y_t is not None:
			feed_dict[self._y_true] = y_t
		if learning_rate is not None:
			feed_dict[self._learning_rate_ph] = learning_rate
			feed_dict[self._learning_rate_decay_ph] = learning_rate_decay
		if self._use_dropout is not None and training:
			feed_dict[self._dropout] = dropout_rate
		else:
			feed_dict[self._dropout] = 1.0
		return feed_dict

	def predict(self, x):
		feed_dict = self.fill_feed_dict(x, training=False)
		if self._use_crf:
			y_pred = []
			logits, trans_params, sequence_lengths = self._sess.run([self._logits,self._trainsition_params,
																	 self._sequence_lengths],feed_dict=feed_dict)
			for logit, sequence_length in zip(logits, sequence_lengths):
				logit = logit[:int(sequence_length)]
				viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
				y_pred += [viterbi_seq]
		else:
			y_pred = self._sess.run(self._y_pred, feed_dict=feed_dict)
		return self.corpus.tag_dict.batch_idxs2batch_toks(y_pred, filter_paddings=True)