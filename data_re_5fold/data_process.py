import pandas as pd 

test_df_sec=pd.read_csv("Second_TestDataSet.csv")
#print(test_df_sec)

test_df_sec['content']=test_df_sec['content'].fillna('无')
test_df_sec['title']=test_df_sec['title'].fillna('无')
print(test_df_sec.info())

submit_example_df = pd.read_csv('submit_example.csv')

print(len(set(submit_example_df['id'])))

submit_id_set = set(submit_example_df['id'])

for i in test_df_sec['id']:
	if i not in submit_id_set:
		print(i, '====')

'''
test_df_sec['label']=0
test_df_sec['content']=test_df_sec['content'].fillna('无')

test_df_sec['title']=test_df_sec['title'].fillna('无')

test_df_sec.to_csv("test.csv")
'''