import pandas as pd
import os
import random



#############初赛数据
#训练集:7340条
#测试集:7356条
######
train_df=pd.read_csv("Train_DataSet.csv")
train_label_df=pd.read_csv("Train_DataSet_Label.csv")
#test_df=pd.read_csv("Test_DataSet.csv")
train_df=train_df.merge(train_label_df,on='id',how='left')
train_df['label']=train_df['label'].fillna(-1)
train_df=train_df[train_df['label']!=-1]
train_df['label']=train_df['label'].astype(int)
#test_df['label']=0

#test_df['content']=test_df['content'].fillna('无')
train_df['content']=train_df['content'].fillna('无')
#test_df['title']=test_df['title'].fillna('无')
train_df['title']=train_df['title'].fillna('无')

#############复赛数据
#训练集:7356条
#测试集:7356条
######
train_df_sec=pd.read_csv("Second_DataSet.csv")
train_label_df_sec=pd.read_csv("Second_DataSet_Label.csv")
test_df_sec=pd.read_csv("Second_TestDataSet.csv")
train_df_sec=train_df_sec.merge(train_label_df_sec,on='id',how='left')
train_df_sec['label']=train_df_sec['label'].fillna(-1)
train_df_sec=train_df_sec[train_df_sec['label']!=-1]
train_df_sec['label']=train_df_sec['label'].astype(int)
test_df_sec['label']=0

test_df_sec['content']=test_df_sec['content'].fillna('无')
train_df_sec['content']=train_df_sec['content'].fillna('无')
test_df_sec['title']=test_df_sec['title'].fillna('无')
train_df_sec['title']=train_df_sec['title'].fillna('无')


#7340+7356=14696   为保证进行10fold,共14695条
train_df = pd.concat([train_df.sample(n=7339, random_state=1), train_df_sec], axis=0)



index=set(range(train_df.shape[0]))
K_fold=[]
k = 5

for i in range(k):
    if i == (k-1):
        tmp=index
    else:
        tmp=random.sample(index,int(1.0/k*train_df.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)
    

for i in range(k):
    print("Fold",i)
    os.system("mkdir data_{}".format(i))
    dev_index=list(K_fold[i])
    train_index=[]
    for j in range(k):
        if j!=i:
            train_index+=K_fold[j]
    train_df.iloc[train_index].to_csv("data_{}/train.csv".format(i))
    train_df.iloc[dev_index].to_csv("data_{}/dev.csv".format(i))
    test_df_sec.to_csv("data_{}/test.csv".format(i))
