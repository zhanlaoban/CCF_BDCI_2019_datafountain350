import pandas as pd
import matplotlib.pyplot as plt 
from collections import Counter


train_df=pd.read_csv("Train_DataSet.csv")
train_label_df=pd.read_csv("Train_DataSet_Label.csv")
test_df=pd.read_csv("Test_DataSet.csv")
train_df=train_df.merge(train_label_df,on='id',how='left')
train_df['label']=train_df['label'].fillna(-1)
train_df=train_df[train_df['label']!=-1]
train_df['label']=train_df['label'].astype(int)
test_df['label']=0

test_df['content']=test_df['content'].fillna('无')
train_df['content']=train_df['content'].fillna('无')
test_df['title']=test_df['title'].fillna('无')
train_df['title']=train_df['title'].fillna('无')

train_df['text'] = train_df['content'] + train_df['title']

#纵坐标为每个文本长度
#横坐标设为[0,7340]
text_len = []
text_len = [ len(i) for i in train_df['text']]
y = sorted(text_len)
#print(y)
print("max ", y[7339])
print("min ", y[0])
x = range(0, 7340)

plt.plot(x, y)
#plt.show()



label_len = [ i for i in train_df['label']]
#print(label_len)
result = Counter(label_len)
plt.figure(figsize=(6,9)) #调节图形大小
#labels = ['大型','中型','小型','微型'] #定义标签
labels = [0, 1, 2]
sizes = [result[0], result[1], result[2]] #每块值
colors = ['red','yellowgreen','lightskyblue','yellow'] #每块颜色定义
#explode = (0,0,0,0) #将某一块分割出来，值越大分割出的间隙越大
patches,text1,text2 = plt.pie(sizes,
 #                     explode=explode,
                      labels=labels,
                      colors=colors,
                      autopct = '%3.2f%%', #数值保留固定小数位
                      shadow = False, #无阴影设置
                      startangle =90, #逆时针起始角度设置
                      pctdistance = 0.6) #数值距圆心半径倍数距离
#patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部的文本
# x，y轴刻度设置一致，保证饼图为圆形
plt.axis('equal')
plt.show()
