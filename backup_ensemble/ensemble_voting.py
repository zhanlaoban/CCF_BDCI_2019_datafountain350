import pandas as pd
import numpy as np

import os



#当模型个数变动时，可以适应
#相对多数投票法算法实现

df = pd.read_csv(os.path.pardir + '/data/submit_example.csv')

#09_26
#models = ['bert_base', 'bert_wwm', 'bert_wwm_ext', 'ERNIE_base', 'roberta_l12', 'roberta_large', 'roberta_wwm_ext']


#09_26_1
#models = ['bert_base', 'bert_wwm', 'ERNIE_base', 'roberta_large']

#09_26_2, 效果达0.81226897000
#models = ['bert_base', 'ERNIE_base', 'roberta_large']

#09_27
#models = ['bert_base', 'bert_wwm', 'roberta_large']

#09_28
#这里的robert-large主要是用了折内进行投票，其单模型效果达到0.80913991000
#这里的ERNIE_base主要是用了折内进行投票，其单模型效果达到0.80636984000
#models = ['bert_base', 'ERNIE_base', 'roberta_large']

#09_29
#基于09_28的基础上,加上bert_wwm_ext的折内投票
#models = ['bert_base', 'ERNIE_base', 'roberta_large', 'bert_wwm_ext']

#10_01
#models = ['backup1_ERNIE_base', 'backup1_roberta_large', 'backup2_roberta_wwm_ext']

#10_02
#models = ['backup1_ERNIE_base', 'backup1_roberta_large', 'backup2_roberta_wwm_ext', 'backup2_roberta_l12']

#10_02_1
#models = ['backup1_ERNIE_base', 'backup1_roberta_large', 'backup2_roberta_wwm_ext', 'backup1_bert_base']

#10_04 以前当票数为[1,1,1]时,默认返回0号,现在设置为返回随机值
models = ['backup1_ERNIE_base', 'backup1_roberta_large', 'backup2_roberta_wwm_ext']



for i in range(len(df)):
	#记票变量
	voting = [0, 0, 0]

	#给每个模型记票
	for model in models:	
		model_sub_dir = '/media/zhan/Mars/datafountain350/results/' + model + '/sub_voting.csv'
		model_df = pd.read_csv(model_sub_dir)

		#df_result['label'] = np.argmax(df_result[['a','b','c']].values, -1)
		vote = model_df['label'][i]
		if vote == 0:
			voting[0] += 1
		elif vote == 1:
			voting[1] += 1
		elif vote == 2:
			voting[2] += 1

	if voting[0]==voting[1]==voting[2]:
		winnerID = np.argmax(np.random.choice([0,1,2]))	
	elif voting[0]==voting[1]>voting[2]:
		winnerID = np.argmax(np.random.choice([0,1]))
	elif voting[1]==voting[2]>voting[0]:
		winnerID = np.argmax(np.random.choice([1,2]))
	elif voting[0]==voting[2]>voting[1]:
		winnerID = np.argmax(np.random.choice([0,2]))
	else:
		winnerID = np.argmax(voting)
	
	df['label'][i] = winnerID

result_path = '/home/zhan/zyy/competitions/datafountain350/ensemble/10_04/10_04_VotingEnsemble.csv'
df.to_csv(result_path, index=False)

