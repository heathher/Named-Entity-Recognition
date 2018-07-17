import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import os

from layers import character_embedding_network, embedding_layer, biLSTM
from evaluation import precision_recall_f1

MODEL_PATH = 'model/'
MODEL_FILE_NAME = 'ner_model.ckpt'

class Network:
	def __init__(self, corpus, n_filters=(128, 128), filter_width=3, token_embeddings_dim=100, char_embeddings_dim=30,
				use_char_embeddins=True, embeddings_dropout=False, use_crf=False, char_filter_width=3, pretrained_model_path=None):

		tf.reset_default_graph()

		n_tags = len(corpus.tag_dict)
		n_tokens = len(corpus.token_dict)
		n_chars = len(corpus.char_dict)

		# Create placeholders
		x_word = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
		x_char = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
		if corpus.embeddings is not None:
			x_emb = tf.placeholder(dtype=tf.float32, shape=[None, None, corpus.emb_size], name='x_word')
		y_true = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')
		mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='mask')
		learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
		dropout_ph = tf.placeholder_with_default(1.0, shape=[])
		training_ph = tf.placeholder_with_default(False, shape=[])
		learning_rate_decay_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_decay')
		momentum_ph = tf.placeholder(dtype=tf.float32, shape=[], name='momentum')
		max_grad_ph = tf.placeholder(dtype=tf.float32, shape=[], name='max_grad')

		# Embeddings
		with tf.variable_scope('Embeddings'):
			w_emb = embedding_layer(x_word, n_tokens=n_tokens, token_embedding_dim=token_embeddings_dim, token_embedding_matrix=corpus.emb_mat)
			w_emb = tf.cast(w_emb, tf.float32)
			#w_emb = tf.to_float(w_emb)
			if use_char_embeddins:
				c_emb = character_embedding_network(x_char, n_characters=n_chars, char_embedding_dim=char_embeddings_dim,
													filter_width=char_filter_width)
				emb = tf.concat([w_emb, c_emb], axis=-1)
			else:
				emb = w_emb
		if corpus.embeddings is not None:
			emb = tf.concat([emb, x_emb], axis=2)
		# Dropout for embeddings
		if embeddings_dropout:
			emb = tf.layers.dropout(emb, dropout_ph, training=training_ph)
		# Make bi-LSTM
		units = biLSTM(emb, n_filters)

		#units = tf.layers.dropout(units, dropout_ph, training=training_ph)

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
		self._learning_rate_decay_ph = learning_rate_decay_ph
		self._x_w = x_word
		self._x_c = x_char
		if corpus.embeddings is not None:
			self._x_emb = x_emb
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
		self.filewriter = tf.summary.FileWriter('graphs', sess.graph)
		self.summary = tf.summary.merge_all()
		self._train_op = self.get_train_op(loss, learning_rate_ph, lr_decay_rate=learning_rate_decay_ph, momentum=momentum_ph, max_grad=max_grad_ph)
		sess.run(tf.global_variables_initializer())
		self._mask = mask
		if pretrained_model_path is not None:
			self.load(pretrained_model_path)
		self._momentum = momentum_ph
		self._max_grad = max_grad_ph



	def get_train_op(self, loss, learning_rate, lr_decay_rate=None, momentum=0.9, max_grad=5.0):
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
		#variables = tf.trainable_variables()
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(extra_update_ops):
			#train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=variables)
			train_op = tf.train.MomentumOptimizer(learning_rate, momentum)
			#for var in tf.all_variables():
			#	print('>', var.name, var.dtype, var.shape)
			gradients, variables = zip(*train_op.compute_gradients(loss))
			gradients, _ = tf.clip_by_global_norm(gradients, max_grad)
			train_op = train_op.apply_gradients(zip(gradients, variables))
		return train_op

	def fit(self, batch_size=10, learning_rate=1e-3, epochs=1, dropout_rate=0.5, learning_rate_decay=1, 
			momentum=0.9, max_grad=5.0):
		for epoch in range(epochs):
			print('Epoch {}'.format(epoch))
			batch_generator = self.corpus.batch_generator(batch_size, dataset_type='train')
			for x, y, token in batch_generator:
				feed_dict = self.fill_feed_dict(x, y, learning_rate, dropout_rate=dropout_rate, training=True,
												 learning_rate_decay=learning_rate_decay, momentum = momentum, max_grad=max_grad)
				#summary, _ = self._sess.run([self.summary, self._train_op], feed_dict=feed_dict)
				#self.filewriter.add_summary(summary)
				self._sess.run(self._train_op, feed_dict=feed_dict)
			self.eval_conll('valid', print_results=True)
			self.save()
		self.eval_conll(dataset_type='train', short_report=False)
		self.eval_conll(dataset_type='valid', short_report=False)
		results = self.eval_conll(dataset_type='test', short_report=False)
		return results

	def eval_conll(self, dataset_type='test', print_results=True, short_report=True):
		y_true_list = list()
		y_pred_list = list()
		file = open('result'+dataset_type+'.txt', 'w')
		print('Eval on {}:'.format(dataset_type))
		for x, y_gt, token in self.corpus.batch_generator(batch_size=10, dataset_type=dataset_type):
			y_pred = self.predict(x)
			y_gt = self.corpus.tag_dict.batch_idxs2batch_toks(y_gt, filter_paddings=True)
			for tags_pred, tags_gt in zip(y_pred, y_gt):
				for tag_predicted, tag_ground_truth in zip(tags_pred, tags_gt):
					y_true_list.append(tag_ground_truth)
					y_pred_list.append(tag_predicted)
				y_true_list.append('O')
				y_pred_list.append('O')
			for tok, y_t, y_p in zip(token, y_gt, y_pred):
				for idx in range(len(tok)):
					file.write("%s ? %s %s\n" %(tok[idx], y_t[idx], y_p[idx]))
				file.write('\n')
		file.close()
		return precision_recall_f1(y_true_list, y_pred_list, print_results, short_report)

	def fill_feed_dict(self, x, y_t=None, learning_rate=None, training=False, dropout_rate=1.0, learning_rate_decay=1.0, 
						momentum=0.9, max_grad=5.0):
		feed_dict = dict()
		# if self.corpus.embeddings is not None:
		# 	feed_dict[self._x_w] = x['emb']
		# else:
		feed_dict[self._x_w] = x['token']
		feed_dict[self._x_c] = x['char']
		feed_dict[self._mask] = x['mask']
		if self.corpus.embeddings is not None:
			feed_dict[self._x_emb] = x['emb']
		feed_dict[self._training_ph] = training
		feed_dict[self._max_grad] = max_grad
		if y_t is not None:
			feed_dict[self._y_true] = y_t
		if learning_rate is not None:
			feed_dict[self._learning_rate_ph] = learning_rate
			feed_dict[self._learning_rate_decay_ph] = learning_rate_decay
			feed_dict[self._momentum] = momentum
		if self._use_dropout is not None and training:
			feed_dict[self._dropout] = dropout_rate
		else:
			feed_dict[self._dropout] = 1.0
		return feed_dict

	def predict(self, x):
		feed_dict = self.fill_feed_dict(x, training=False)
		if self._use_crf:
			y_pred = []
			logits, trans_params, sequence_lengths = self._sess.run([self._logits, self._trainsition_params,
																	 self._sequence_lengths],feed_dict=feed_dict)
			for logit, sequence_length in zip(logits, sequence_lengths):
				logit = logit[:int(sequence_length)]
				viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
				y_pred += [viterbi_seq]
		else:
			y_pred = self._sess.run(self._y_pred, feed_dict=feed_dict)
		return self.corpus.tag_dict.batch_idxs2batch_toks(y_pred, filter_paddings=True)

	def load(self, model_file_path):
		saver = tf.train.Saver()
		saver.restore(self._sess, model_file_path)
			
	def save(self, model_file_path=None):
		if model_file_path is None:
			if not os.path.exists(MODEL_PATH):
				os.mkdir(MODEL_PATH)
			model_file_path = os.path.join(MODEL_PATH, MODEL_FILE_NAME)
		saver = tf.train.Saver()
		saver.save(self._sess, model_file_path)

	
