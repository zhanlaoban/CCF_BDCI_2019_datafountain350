import pandas as pd
import numpy as np

import os



#当模型个数变动时，可以适应
#相对多数投票法算法实现

df = pd.read_csv(os.path.pardir + '/data/submit_example.csv')

#10_28_1 权重分布为4,3,2 
#0.81947172
#models = ['backu13_ERNIE_base', 'backu11_bert_base', 'backup6_roberta_wwm_large_ext']

#10_28_2 权重分布为4,3,2 
#0.81857151000
#models = ['backu13_ERNIE_base', 'backup2_xlnet_mid', 'backup6_roberta_wwm_large_ext']

#10_28_3 权重分布为4,3,2 
#0.81237835000
#models = ['backu17_ERNIE_base', 'backup7_bert_base', 'backup4_roberta_wwm_large_ext']

#10_28_4 权重分布为4,3,2 
#0.82043248000
#models = ['backu17_ERNIE_base', 'backup2_xlnet_mid', 'backup6_roberta_wwm_large_ext']

#10_28_5 权重分布为5,2,2,2
#0.81930119000
#models = ['backup2_xlnet_mid', 'backu17_ERNIE_base', 'backu11_bert_base', 'backup6_roberta_wwm_large_ext']

#10_30_1 权重分布为4,3,2 
#
models = ['backu17_ERNIE_base', 'backup2_xlnet_mid', 'backup7_roberta_wwm_large_ext']

for i in range(len(df)):
	#记票变量
	voting = [0, 0, 0]
	w1 = 4
	w2 = 3
	w3 = 2

	#给每个模型记票
	for model in models:
		
		if model == 'backu11_bert_base':
			model_sub_dir = '/media/zhan/Mars/datafountain350/results/' + model + '/sub_averaging.csv'
			model_df = pd.read_csv(model_sub_dir)
		else:
			model_sub_dir = '/media/zhan/Mars/datafountain350/results/' + model + '/sub_voting.csv'
			model_df = pd.read_csv(model_sub_dir)
			#model_df['label'] = np.argmax(model_df[['label_0','label_1','label_2']].values, -1)
		

		#df_result['label'] = np.argmax(df_result[['a','b','c']].values, -1)
		#model_sub_dir = '/media/zhan/Mars/datafountain350/results/' + model + '/sub_voting.csv'
		#model_df = pd.read_csv(model_sub_dir)
		vote = model_df['label'][i]
		if model == 'backup7_roberta_wwm_large_ext':			
			if vote == 0:
				voting[0] += w1
			elif vote == 1:
				voting[1] += w1
			elif vote == 2:
				voting[2] += w1

		if model == 'backup2_xlnet_mid':
			if vote == 0:
				voting[0] += w2
			elif vote == 1:
				voting[1] += w2
			elif vote == 2:
				voting[2] += w2

		
		if model == 'backu17_ERNIE_base':
			if vote == 0:
				voting[0] += w3
			elif vote == 1:
				voting[1] += w3
			elif vote == 2:
				voting[2] += w3

		
		

	winnerID = np.argmax(voting)
	df['label'][i] = winnerID

result_path = '/home/zhan/zyy/competitions/datafountain350/ensemble/10_30/10_30_1_WeightedVotingEnsemble.csv'
df.to_csv(result_path, index=False)

