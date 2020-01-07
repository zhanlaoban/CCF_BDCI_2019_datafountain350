import pandas as pd
import numpy as np

import os


#10_19_1 求平均
models = ['backup1_ERNIE_base', 'backup5_roberta_large', 'backup1_roberta_wwm_large_ext']


#本函数将结果合成一个csv文件
def generate_result():
	df = pd.read_csv(os.path.pardir + '/data/submit_example.csv')
	df['0']=0
	df['1']=0
	df['2']=0

	for model in models:
		#生成模型路径和输出路径
		model_prefix = '/media/zhan/Mars/datafountain350' + '/results/' + model + '/' + model.replace(model[:8], '')
		out_path = '/media/zhan/Mars/datafountain350' + '/results/' + model + '/' + 'prob_' + model.replace(model[:8], '') + '.csv'

		for i in range(5):
			temp = pd.read_csv('{}{}/sub.csv'.format(model_prefix, i))
			df['0'] += temp['label_0']
			df['1'] += temp['label_1']
			df['2'] += temp['label_2']

		df[['id','0', '1', '2']].to_csv(out_path, index=False)



#Averaging：概率等权重融合
#对bert、roberta-large、xlnet进行融合
#上述每个模型产生一个三维的向量，每个向量表示选择对应标签的概率
#将每个模型的这个三维向量相加，最后使用argmax函数


def AveragingEnsemble():
	result_path = os.path.pardir + '/ensemble/10_19/10_19_1_AveragingEnsemble.csv'

	df_result = pd.read_csv(os.path.pardir + '/data/submit_example.csv')
	df_result['a']=0
	df_result['b']=0
	df_result['c']=0

	for model in models:
		prob_csv_dir = '/media/zhan/Mars/datafountain350' + '/results/' + model + '/' + 'prob_' + model.replace(model[:8], '') + '.csv'
		temp = pd.read_csv(prob_csv_dir)
		df_result['a'] += temp['0']
		df_result['b'] += temp['1']
		df_result['c'] += temp['2']

	df_result['label'] = np.argmax(df_result[['a','b','c']].values, -1)
	df_result[['id','label']].to_csv(result_path, index=False)


#代码思路:
#1.先生成pro_model.csv文件
#2.在1的基础上求平均
#generate_result()
AveragingEnsemble()
