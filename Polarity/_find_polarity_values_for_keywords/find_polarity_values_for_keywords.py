import json
import pandas as pd

from ._get_polarity_of_contents_for_word import _get_polarity_of_contents_for_word

from DataLoader import load_polarity_file
from DataLoader import save_polarity_to_file

def find_polarity_values_for_keywords(df, **params):
	
	important_keywrods = {
		"epidemic":["疫情","炎防疫","疫情","瘟疫","瘟","暴發","發作"],
		"quarantine":["檢疫","檢疫","隔離"],
		"Hang Seng":["恆生","股票","庫存","存貨","采購","市場","銷售",
						"集市","推銷","投資者","投資","花費","給"],
		"bank":["銀行","岸","岸邊"],
		"financial":["財政","經濟","金融","金融界","經濟的","經濟",
						"投資","投資額","商業","業務","商"],
		"government":["政府","政府","內閣","轄"],
		"hospital":["醫院","治療","待遇","對待"],
		"china":["瓷器","瓷","瓷的","北京","京","上海","深圳","香港",
						"國家的","國民","國","國立","公民","武漢","中國人",
						"中國","中華","中華","內地","大陸"],
		"policy":["政策","方針","筴","總統","主席","校長","副主席",
						"政治的","政治","政","政見"],
		"vaccine":["疫苗","苗","痘","科興","生物科技","劑量","服用量",
						"服葯","疫苗接種"],
		"mask":["面具","假面具","鬼臉","隱藏","面具"],
		"covid":["冠狀病毒病","電暈","病毒","細菌","惡毒","感染",
						"傳染","肺炎","肺病"],
		"medical":["醫療的","醫","醫生","健康","衛生","身體","醫生",
							"博士","行醫","藥品","醫學","藥","醫藥"],
		"wave":["海浪","波","浪","波動"],
		"social":["社會的","群居","社交的","人們","萌"],
		}

	# important_keywrods = {
	# 	"epidemic":["疫情","炎防疫","疫情","瘟疫","瘟","暴發","發作"],
	# 	"quarantine":["檢疫","檢疫"],
	# 	"Hang Seng":["恆生","股票"],
	# 	}

	tmp_df = df.copy()
	for eng_word, ls_words in important_keywrods.items():

		polarity_file = load_polarity_file()

		if isinstance(polarity_file, pd.DataFrame):
			df = polarity_file
			if eng_word in list(polarity_file.columns):
				print (f"Polarity of --->{eng_word}<---- is already analyzed")
				continue

		print (f"Polarity of --->{eng_word}<---- is about to be analyzed")
		df[eng_word] = \
			_get_polarity_of_contents_for_word(tmp_df, ls_words, **params)
		save_polarity_to_file(df)


	return