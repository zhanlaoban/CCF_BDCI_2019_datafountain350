import pandas as pd
import numpy as np
import os

#5fold best:
#backup1_bert_base, backup1_bert_wwm, backup2_bert_wwm_ext, backup1_ERNIE_base, backup2_roberta_l12
#backup1_roberta_large, backup2_roberta_wwm_ext

#10fold

models = ['backup7_bert_base',
		  'backup2_bert_wwm',	
		  'backup3_bert_wwm_ext',
		  'backup3_ERNIE_base',
		  'backup1_xlnet_mid',	
		  'backup5_roberta_large',
		  'backup1_roberta_wwm_large_ext'
		  ]


#10fold
'''
models = ['backup2_bert_base',
		  'backup3_ERNIE_base',
		  'backup5_roberta_large'
		  ]
'''

#5fold
'''
models = ['backup1_bert_base',
		  'backup1_bert_wwm',
		  'backup2_bert_wwm_ext',
		  'backup1_ERNIE_base',
		  'backup1_roberta_large',
		  'backup2_roberta_wwm_ext']
'''

models_dir = '/media/zhan/Mars/datafountain350/results/'

k = 10
model_num = 7

if k == 5:
	dev_dir = '../data'
elif k == 10:
	dev_dir = '../data_10fold'

#1.读取data/data_i/dev.csv,将0~4的进行按序合并,放在data/Dev.csv下
def generateDev():
	df_list = []
	

	for i in range(k):
		df_list.append(pd.read_csv(dev_dir + '/data_{}/dev.csv'.format(i)))

	df_res = pd.concat(df_list, axis=0)
	print(df_res)
	df_res[['id','label']].to_csv(dev_dir + '/Dev.csv', index=False)

#generateDev()
#现已生成,无需再单独生成

#2.将一些没有predicted_dev.csv的模型,生成其predicted_dev.csv
#models_genedev = ['backup1_bert_base', 'backup1_ERNIE_base', 'backup1_roberta_large']
#现已生成


#3.读取/media/zhan/Mars/datafountain350/results/backup1_bert_base/bert_basei/predicted_dev.csv
#将0~4的进行按序合并.然后将每个Model的predicted_dev.csv合并,成一个7340*n的dataframe,n为模型个数
def generate_predicted_Dev():
	DF_list = []
	for model in models:
		df_list = []
		model_dir = models_dir + model + '/'
		#print(model_dir)

		for i in range(k):	
			csv_dir = (model_dir + model.replace(model[:8], '') + '{}/predicted_dev.csv'.format(i))
			#print(csv_dir)
			df_list.append(pd.read_csv(csv_dir))


		df_res = pd.concat(df_list, axis=0)
		model_name = model.replace(model[:8], '')
		#df_res.columns = [model_name + '_0', model_name + '_1', model_name + '_2']
		df_res.drop(columns='id', inplace=True)
		df_res.rename(columns = {df_res.columns[0]:model_name + '_0', df_res.columns[1]:model_name + '_1', df_res.columns[2]:model_name + '_2',}, inplace=True)
		DF_list.append(df_res)

	#将所有的model合并
	allmodel_predicted_Dev = pd.concat(DF_list, axis=1)
	#print(len(allmodel_predicted_Dev))
	#print(allmodel_predicted_Dev.columns.values)

	#将Dev.csv和allmodel_predicted_Dev合并
	df_Dev = pd.read_csv(dev_dir + '/Dev.csv')
	df_Dev.drop(columns='id', inplace=True)
	

	print(len(allmodel_predicted_Dev))
	print("meta_feature_train == ", len(allmodel_predicted_Dev.columns.values))

	allmodel_predicted_Dev.to_csv('./stacking/' + str(k) + 'fold/' + str(model_num) + 'models/meta_feature_train.csv', index=False)
	df_Dev.to_csv('./stacking/' + str(k) + 'fold/' + str(model_num) + 'models/meta_feature_train_label.csv', index=False)

		#print(len(df_res))
		#print(df_res.columns.values)
		#print(df_res)

		#df_res.to_csv(model_dir + 'predicted_Dev.csv', index=False)

generate_predicted_Dev()
#现已生成


###########################
#前面都是针对 验证集 的预处理
#下面是针对 测试集 的预处理
###########################
def generate_test_meta_feature():
	

	df_list = []

	for model in models:
		model_dir = models_dir + model + '/'
		df_res = pd.read_csv(model_dir + 'sub_averaging.csv')
		print(model)
		
		
		
		df_res.drop(columns='id', inplace=True)
		model_name = model.replace(model[:8], '')
		print(df_res.columns.values)
		#print(df_res.columns[0], df_res.columns[1], df_res.columns[2])
		df_res.rename(columns = {df_res.columns[0]:model_name + '_0', df_res.columns[1]:model_name + '_1', df_res.columns[2]:model_name + '_2',}, inplace=True)

		df_list.append(df_res)

	DF_res = pd.concat(df_list, axis=1)


	#print(len(DF_res))
	print(DF_res.columns.values)
	print("meta_feature_test == ", len(DF_res.columns.values))

	DF_res.to_csv('./stacking/' + str(k) + 'fold/' + str(model_num) + 'models/meta_feature_test.csv', index=False)

		#print(model_dir)
		#csv_dir = (model_dir + model.replace(model[:8], '') + '{}/predicted_dev.csv'.format(i))
		#print(csv_dir)
		#df_list.append(pd.read_csv(csv_dir))



generate_test_meta_feature()