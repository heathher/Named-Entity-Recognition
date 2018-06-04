import tensorflow as tf
import numpy as np

def biLSTM(input_units, n_hidden_list):
	units = input_units
	for n, n_h in enumerate(n_hidden_list):
		with tf.variable_scope('LSTM' + str(n)):
			forward_cell = tf.nn.rnn_cell.LSTMCell(n_h)
			backward_cell = tf.nn.rnn_cell.LSTMCell(n_h)
			(rnn_output_fw, rnn_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, units,dtype=tf.float32)
			units = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)
	return units

def embedding_layer(input_placeholder=None, token_embedding_matrix=None, n_tokens=None,
					token_embedding_dim=None, name=None):
	tok_mat = np.random.randn(n_tokens, token_embedding_dim).astype(np.float32) / np.sqrt(token_embedding_dim)
	tok_emb_mat = tf.Variable(tok_mat, trainable=True)
	embeddings = tf.nn.embedding_lookup(tok_emb_mat, input_placeholder)
	return embeddings

def character_embedding_network(char_placeholder, n_characters, char_embedding_dim, filter_width=7):
	char_emb_mat = np.random.randn(n_characters, char_embedding_dim).astype(np.float32) / np.sqrt(char_embedding_dim)
	char_emb_var = tf.Variable(char_emb_mat, trainable=True)
	with tf.variable_scope('CharEmbNetwork'):
		c_emb = tf.nn.embedding_lookup(char_emb_var, char_placeholder)
		char_conv = tf.layers.conv2d(c_emb, char_embedding_dim, (1, filter_width), padding='same', name='char_conv')
		char_emb = tf.reduce_max(char_conv, axis=2)
	return char_emb