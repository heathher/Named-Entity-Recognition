import os
import re
from collections import namedtuple
from ufal.udpipe import Model, Pipeline

FactRu = namedtuple('FactRu', ['id', 'text'])


def tag_ud(text='Текст нужно передать функции в виде строки!', modelfile='udpipe_syntagrus.model'):
	model = Model.load(modelfile)
	pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
	processed = pipeline.process(text) # обрабатываем текст, получаем результат в формате conllu
	output = [l for l in processed.split('\n') if not l.startswith('#')] # пропускаем строки со служебной информацией
	tagged = [w.split('\t')[2].lower() + '_' + w.split('\t')[3] for w in output if w] # извлекаем из обработанного текста лемму и тэг
	tagged_propn = []
	propn  = []
	for t in tagged:
		if t.endswith('PROPN'):
			if propn:
				propn.append(t)
			else:
				propn = [t]
		else:
			if len(propn) > 1:
				for x in propn:
					#name = '::'.join([x.split('_')[0] for x in propn]) + '_PROPN'
					tagged_propn.append(x)
			elif len(propn) == 1:
				tagged_propn.append(propn[0])
			tagged_propn.append(t)
			propn = []
	return tagged_propn

def read_dataset(dir_path, filetype):
	for filename in os.listdir(dir_path):
		match = re.match('book_(\d+)\.'+filetype, filename)
		if match:
			id = int(match.group(1))
			path = os.path.join(dir_path, filename)
			with open(path) as f:
				text = list()
				for line in f:
					items = line.split()
					if len(items)>1:
						if filetype == 'tokens':
							text.append(items[3])
						else:
							entity = list()
							entity.append(items[1])
							counter = 0
							while items[counter+2] != '#':
								counter += 1
							ind_begin = counter+3
							while counter > 0:
								entity.append(items[ind_begin])
								counter -= 1
								ind_begin += 1
							text.append(entity)
			yield FactRu(id, text)

def make_xy(tokens, objects):
	xy_list = list()
	tokens_list = list()
	tags_list = list()
	new_tokens_list = list()
	for i in range(len(tokens)):
		tokens_list_for_id = tokens[i]
		tags_list_for_id = [tags for tags in objects if tokens_list_for_id.id == tags.id]
		tokens_list_for_id_text = tokens_list_for_id.text
		tags_list_for_id_text = tags_list_for_id[0].text
		prev_tag = None
		tag = None
		for ind in range(len(tokens_list_for_id_text)):
			token = tokens_list_for_id_text[ind]
			
			for tags in tags_list_for_id_text:
				match = [word for word in tags if word == token]
				if len(match) > 0:
					tag = tags[0]
			if tag == None:
				tag = 'O'
			elif tag == 'Person':
				tag = 'PER'
			elif tag == 'Org':
				tag = 'ORG'
			elif tag == 'Location' or tag =='LocOrg':
				tag = 'LOC'
			else:
				tag = 'MISC'			
			if tag == prev_tag and tag != 'O':
				prev_tag = tag
				tag = 'I-' + tag
			elif tag != 'O':
				prev_tag = tag
				tag = 'B-' + tag
			else:
				prev_tag = tag
			tokens_list.append(token)
			tags_list.append(tag)
			tag = None
			if token == '.':
				tokens_string = ''
				for token in tokens_list:
					tokens_string += token
					tokens_string += ' '
				new_tokens_list = tag_ud(tokens_string, 'rus_emb/'+'udpipe_syntagrus.model')
				#print((new_tokens_list, tags_list, ))
				xy_list.append((new_tokens_list, tags_list,))
				tokens_list = list()
				new_tokens_list = list()
				tags_list = list()
	return xy_list

def write_to_file(dataset_type, dataset):
	file = open(dataset_type+'.txt', 'w')
	for token, tag in dataset[dataset_type]:
		for idx in range(len(token)):
			file.write("%s %s\n" %(token[idx], tag[idx]))
		file.write('\n')
	file.close()