import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix", default=None, type=str, required=True)
parser.add_argument("--out_path", default=None, type=str, required=True)
args = parser.parse_args()

k = 10
df = pd.read_csv('data_re_5fold/data_0/test.csv')

def kfold_averaging():
	
	df['0']=0
	df['1']=0
	df['2']=0
	for i in range(k):
	    temp=pd.read_csv('{}{}/sub.csv'.format(args.model_prefix,i))

	    #将某一个标签上的概率值相加取平均
	    df['0']+=temp['label_0']/k
	    df['1']+=temp['label_1']/k
	    df['2']+=temp['label_2']/k
	#print(df['0'].mean())
	 
	df['label']=np.argmax(df[['0','1','2']].values,-1)
	df[['id','label']].to_csv(args.out_path,index=False)

	#df[['id','0', '1', '2']].to_csv(args.out_path,index=False)


def kfold_voting():
	for i in range(len(df)):
		#记票变量
		voting = [0, 0, 0]

		#给每个fold模型记票
		for z in range(k):	
			temp = pd.read_csv('{}{}/sub.csv'.format(args.model_prefix,z))

			temp['label'] = np.argmax(temp[['label_0','label_1','label_2']].values, -1)
			vote = temp['label'][i]
			
			if vote == 0:
				voting[0] += 1
			elif vote == 1:
				voting[1] += 1
			elif vote == 2:
				voting[2] += 1

		winnerID = np.argmax(voting)
		df['label'][i] = winnerID

	df[['id','label']].to_csv(args.out_path,index=False)

#kfold_averaging()

kfold_voting()


#bert_base
#python combine.py --model_prefix ./results/bert_base_02/bert_base --out_path ./results/bert_base_02/sub_voting.csv
#python combine.py --model_prefix ./results/bert_base_04/bert_base --out_path ./results/bert_base_04/sub_averaging.csv

#bert_wwm
#python combine.py --model_prefix /media/zhan/Mars/datafountain350/results/backup2_bert_wwm/bert_wwm --out_path /media/zhan/Mars/datafountain350/results/backup2_bert_wwm/sub_averaging.csv
#python combine.py --model_prefix ./results/bert_wwm/bert_wwm --out_path ./results/bert_wwm/sub_voting.csv

#bert_wwm_ext
#python combine.py --model_prefix ./results/bert_wwm_ext_03/bert_wwm_ext --out_path ./results/bert_wwm_ext_03/sub_voting.csv

#ERNIE_base
#python combine.py --model_prefix ./results/ERNIE_base_06/ERNIE_base --out_path ./results/ERNIE_base_06/sub_voting.csv

#OpenCLaP_baike
#python combine.py --model_prefix ./results/OpenCLaP_baike/OpenCLaP_baike --out_path ./results/OpenCLaP_baike/sub.csv

#roberta_l12
#python combine.py --model_prefix ./results/roberta_l12/roberta_l12 --out_path ./results/roberta_l12/sub.csv
#python combine.py --model_prefix ./results/roberta_l12/roberta_l12 --out_path ./results/roberta_l12/sub_voting.csv

#roberta_large
#python combine.py --model_prefix ./results/backup1_roberta_large/roberta_large --out_path ./results/backup1_roberta_large/sub_voting_3best.csv
#python combine.py --model_prefix ./results/backup2_roberta_large/roberta_large --out_path ./results/backup2_roberta_large/sub_averaging.csv
#python combine.py --model_prefix ./results/backup4_roberta_large/roberta_large --out_path ./results/backup4_roberta_large/sub_voting.csv
#python combine.py --model_prefix ./results/backup3_roberta_large/roberta_large --out_path ./results/backup4_roberta_large/sub_averaging.csv
#python combine.py --model_prefix ./results/backup4_roberta_large/roberta_large --out_path ./results/backup4_roberta_large/sub_averaging.csv
#python combine.py --model_prefix ./results/backup6_roberta_large/roberta_large --out_path ./results/backup6_roberta_large/sub_voting.csv
#python combine.py --model_prefix /media/zhan/Mars/datafountain350/results/backup5_roberta_large/roberta_large --out_path /media/zhan/Mars/datafountain350/results/backup5_roberta_large/sub_voting.csv



#roberta_wwm_large_ext
#python combine.py --model_prefix ./results/roberta_wwm_large_ext_04/roberta_wwm_large_ext --out_path ./results/roberta_wwm_large_ext_04/sub_voting.csv


#xlnet_mid
#python combine.py --model_prefix ./results/xlnet_mid_02/xlnet_mid --out_path ./results/xlnet_mid_02/sub.csv

#stacking
#python combine.py --model_prefix ./ensemble/stacking/results/bertFC_base/bertFC_base --out_path ./ensemble/stacking/results/bertFC_base/stacking_voting.csv