import pandas as pd
import numpy as np

import os



#当模型个数变动时，可以适应
#相对多数投票法算法实现

df = pd.read_csv(os.path.pardir + '/data_re_5fold/data_0/test.csv')


#11_16_1 权重分布为5,2,2,2
#0.81428820000
#models = ['roberta_wwm_large_ext_03', 'bert_base_04', 'bert_wwm_ext_03', 'xlnet_mid_02']

#11_16_2 权重分布为4,3,2
#0.81567681000 
#models = ['roberta_wwm_large_ext_03', 'bert_wwm_ext_03', 'xlnet_mid_02']

#11_16_3 权重分布为4,3,2
#0.81318772000
#models = ['roberta_wwm_large_ext_03', 'bert_base_04', 'xlnet_mid_02']

#11_18_1 权重分布为4,3,2
#
models = ['roberta_wwm_large_ext_04', 'bert_wwm_ext_03', 'xlnet_mid_02']

for i in range(len(df)):
	#记票变量
	voting = [0, 0, 0]
	w1 = 4
	w2 = 3
	w3 = 2


	#给每个模型记票
	for model in models:
		
		if model == 'bert_base_04':
			#model_sub_dir = '/media/zhan/Mars/datafountain350/results/' + model + '/sub_averaging.csv'
			model_sub_dir = '../results/' + model + '/sub_averaging.csv'
			model_df = pd.read_csv(model_sub_dir)
		else:
			model_sub_dir = '../results/' + model + '/sub_voting.csv'
			model_df = pd.read_csv(model_sub_dir)
			#model_df['label'] = np.argmax(model_df[['label_0','label_1','label_2']].values, -1)
		

		#df_result['label'] = np.argmax(df_result[['a','b','c']].values, -1)
		#model_sub_dir = '/media/zhan/Mars/datafountain350/results/' + model + '/sub_voting.csv'
		#model_df = pd.read_csv(model_sub_dir)
		vote = model_df['label'][i]
		if model == 'roberta_wwm_large_ext_04':			
			if vote == 0:
				voting[0] += w1
			elif vote == 1:
				voting[1] += w1
			elif vote == 2:
				voting[2] += w1

		if model == 'xlnet_mid_02':
			if vote == 0:
				voting[0] += w2
			elif vote == 1:
				voting[1] += w2
			elif vote == 2:
				voting[2] += w2

		
		if model == 'bert_wwm_ext_03':
			if vote == 0:
				voting[0] += w3
			elif vote == 1:
				voting[1] += w3
			elif vote == 2:
				voting[2] += w3

		
		
		

	winnerID = np.argmax(voting)
	df['label'][i] = winnerID

result_path = '/home/zhan/zyy/competitions/datafountain350/ensemble/11_18/11_18_1.csv'
df[['id','label']].to_csv(result_path, index=False)

