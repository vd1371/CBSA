import pickle
import json

import fasttext
import fasttext.util

from DataLoader import load_unique_words

# !wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
# import zipfile

# target = 'v0.9.2.zip'

# handle = zipfile.ZipFile(target)
# handle.extractall()
# handle.close()
# cd ../fastText-0.9.2

def pkl_embedding(**params):
	unique_words = load_unique_words()

	word_embedding = {300:{}, 200:{}, 100:{}}

	base_dir = "./embedding/_fasttext/"

	for emb_dimension in [300]:

		ft = fasttext.load_model(base_dir + "cc.zh.300.bin")

		print (f"Trying to do fastText embedding for dim:{emb_dimension}")

		if emb_dimension in [200, 100]:
			fasttext.util.reduce_model(ft, emb_dimension)

		for word in unique_words:
			word_embedding[emb_dimension][word] = ft.get_word_vector(word)

		with open(base_dir + \
					f'fasttext_embedding_{emb_dimension}_pretrained.pkl',
						'wb') as f:
			pickle.dump(word_embedding, f)
