#由于无法通过命令行端fq,故本代码在colab上运行的

from joblib import Parallel, delayed
#from textblob import TextBlob
#from textblob.translate import NotTranslated

from translate import Translator

import argparse
import os
import pandas as pd


def gene_df():
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

    return train_df, test_df


NAN_WORD = "_NAN_"



def translate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    #text = TextBlob(comment)
    
    try:
        #text = text.translate(to=language)
        #text = text.translate(to="zh-CN")
        translator_en = Translator(from_lang="zh", to_lang="en")
        translation_en = translator_en.translate(comment)

        translator_zh = Translator(to_lang="zh")
        translation_zh = translator_zh.translate(translation)
    except:
        pass

    return str(translation)

'''
def translate(comment, language):
    return str(comment) + "[][]"
'''

def convert(train_df, filename, args):
    
        #comments_list = train_data["comment_text"].fillna(NAN_WORD).values

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    parallel = Parallel(args.thread_count, backend="threading", verbose=5)
    for language in args.languages:
        train_data = train_df.copy()
        print('Translate comments using "{0}" language'.format(language))

        for text in ['title', 'content']:
            translated_data = []
            comments_list = train_data[text]
            translated_data = parallel(delayed(translate)(comment, language) for comment in comments_list)
            #translated_data = [(translate)(comment, language) for comment in comments_list]
            #for comment in comments_list:
            #    translated_data.append((translate)(comment, language))

            train_data[text] = translated_data

        result_path = os.path.join(args.result_path, filename + language + ".csv")
        train_data.to_csv(result_path, index=False)

def main():
    parser = argparse.ArgumentParser("Script for extending train dataset")
    #parser.add_argument("train_file_path")
    #parser.add_argument("--languages", nargs="+", default=["es", "de", "fr", "en"])
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--thread-count", type=int, default=300)
    parser.add_argument("--result-path", default="extended_data")

    args = parser.parse_args()

    train_df, test_df = gene_df()

    convert(train_df, 'train_', args)
    convert(test_df, 'test_', args)
    


if __name__ == "__main__":
    main()


#python extend_dataset.py