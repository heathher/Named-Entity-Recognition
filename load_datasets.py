import os
import re
from collections import namedtuple


FactRu = namedtuple('FactRu', ['id', 'text'])

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
	for i in range(len(tokens)):
		tokens_list_for_id = tokens[i]
		tags_list_for_id = [tags for tags in objects if tokens_list_for_id.id == tags.id]
		tokens_list_for_id_text = tokens_list_for_id.text
		tags_list_for_id_text = tags_list_for_id[0].text
		prev_tag = None
		tag = None
		for ind in range(len(tokens_list_for_id_text)):
			token = tokens_list_for_id_text[ind]
			tokens_list.append(token)
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
			elif tag == 'Location':
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
			tags_list.append(tag)
			tag = None
			if token == '.':
				xy_list.append((tokens_list, tags_list))
				tokens_list = list()
				tags_list = list()
	return xy_list
