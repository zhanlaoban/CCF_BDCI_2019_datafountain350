from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated

import argparse
import os
import pandas as pd


#1.预处理成7340条
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

#2.提取出0号标签的数据,将其分别进行5种不同语言的转换翻译,title和content分别进行转换
#3.提取出2号标签的数据,将其分别进行5种不同语言的转换翻译,只增加763条即可,title和content分别进行转换


NAN_WORD = "_NAN_"



def translate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    text = TextBlob(comment)
    try:
        text = text.translate(to=language)
        text = text.translate(to="en")
    except NotTranslated:
        pass

    return str(text)


def translate_label0():



def main():
    parser = argparse.ArgumentParser("Script for extending train dataset")
    parser.add_argument("train_file_path")
    parser.add_argument("--languages", nargs="+", default=["es", "de", "fr"])
    parser.add_argument("--thread-count", type=int, default=300)
    parser.add_argument("--result-path", default="extended_data")

    args = parser.parse_args()

    train_data = pd.read_csv(args.train_file_path)
    comments_list = train_data["comment_text"].fillna(NAN_WORD).values

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    parallel = Parallel(args.thread_count, backend="threading", verbose=5)
    for language in args.languages:
        print('Translate comments using "{0}" language'.format(language))
        translated_data = parallel(delayed(translate)(comment, language) for comment in comments_list)
        train_data["comment_text"] = translated_data

        result_path = os.path.join(args.result_path, "train_" + language + ".csv")
        train_data.to_csv(result_path, index=False)


if __name__ == "__main__":
    main()


#python extend_dataset.py train_label_0.csv