import re
import pandas as pd
import fasttext

from utils import ParallelProcess

def _drop_non_chinese(df, **params):

	print ("Trying to _drop_non_chinese")
	n_cores = params.get("n_cores")

	results = ParallelProcess(df,
								_drop_non_chinese_for_parallel,
								n_cores = n_cores)

	return results

def _drop_non_chinese_for_parallel(df):

	model = fasttext.load_model('./DataLoader/lid.176.ftz')

	mask = []
	new_content = []
	for content in df['content']:
		# print (content)
		# lang = chardet.detect(content)
		new_content = ""
		for char in re.findall(r'[\u4e00-\u9fff]+', content):
			new_content += char
		lang = model.predict(new_content.replace('\n', ""), k=1)[0][0]

		if lang == "__label__zh":
			mask.append(True)
		else:
			mask.append(False)

	df = df[mask]

	return df