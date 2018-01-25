# Filename: email_classify.py
# Date:     2018/01/25
# Author:   Bing

# coding: utf-8
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from math import log10
from os import listdir

# def cleansing: remove the punctuation and stopwords   
# input: string text
# output: string clean_text
def cleansing(text):
    clean_text = re.sub(r'[!?/\\"-:=$_]','',text)
    clean_text = re.sub(r'[<||>]',' ',clean_text)
    clean_text = re.sub(r'@','',clean_text)
    clean_text = re.sub(r'\n',' ',clean_text)
    clean_text = re.sub(r' +',' ',clean_text)

    en_stop = get_stop_words('en')
    clean_text = ' '.join([word for word in clean_text.lower().split() if not word in en_stop])
    return clean_text

# def getMsg: catch the content after the first newline and clean it.   
# input: string path
# output: string msg
def getMsg(path):
    msg_list = []
    flag = False
    with open(path, encoding="latin1") as f:
        doc = f.readlines()
    for row in doc:
        if row == '\n':
            flag = True
        if flag:
            msg_list.append(row)
    msg = cleansing(" ".join(msg_list))
    return msg

# def getTdm: calculate the term frequency and return a term document matrix
# input: list doc_vec, int min_freq
# output: daframe tdm
def getTdm(doc_vec, min_freq = 2):
    try:
        vec = CountVectorizer(min_df = min_freq)
        X = vec.fit_transform(doc_vec)
        tdm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
        return tdm
    except:
        return pd.DataFrame(columns=['term'])

# def classify: calculate the probability that email is a spam(or normal)
# input: string path, dataframe training, int prior
# output: float result
def classify(path, training, prior = 0.5, c = 1e-6):
    msg = getMsg(path)
    tdm = getTdm([msg], min_freq = 1)
    if len(tdm) == 0:
        return 0
    match = pd.Series(list(set(tdm.columns).intersection(set(training['term']))))
    if len(match) < 1:
        result = log10(prior) + (log10(c) * len(tdm.columns))
        return result
    else:
        probs = training[training['term'].isin(match)]
        result = log10(prior) + (probs['occurrence'].apply(lambda x:log10(x)).sum()) + (log10(c) * (len(tdm.columns) - len(match)))
        return result
# spamClassifier: predict the email whether is a spam email or not . 
# input: string path
# output: boolean
def spamClassifier(path):
    spam = pd.read_csv('data/spam.csv',encoding ='latin1')
    ham = pd.read_csv('data/easyham.csv',encoding ='latin1')
    spamtest = classify(path, training = spam, prior = 0.2, c = 1e-6)
    hamtest = classify(path, training = ham, prior = 0.8, c = 1e-6)
    return spamtest >= hamtest

# training
def training():
    global spam_path, easyham_path
    
    # training spam data
    spam_dir = listdir(spam_path)
    all_spam = [getMsg(spam_path+file) for file in spam_dir]
    tdm_spam = getTdm(all_spam)

    # caculate frequency, density, occurrence
    frequency = tdm_spam.sum(axis=0)
    density = tdm_spam.apply(lambda col: col.sum()/frequency.sum(), axis=0)
    occurrence = tdm_spam.apply(lambda col: (col != 0).sum()/col.count(), axis=0)
    spam = pd.concat([frequency, density, occurrence], axis=1, join_axes=[frequency.index])
    spam.columns = ['frequency','density','occurrence']
    spam['term'] = spam.index
    spam = spam[['term', 'frequency','density','occurrence']]
    spam = spam.sort_values(by='occurrence', ascending=False)

    # save to csv file
    spam.to_csv("data/spam.csv", sep=',', encoding='latin1', index=False)

    # training easyham data, do the same thing like training spam data
    easyham_dir = listdir(easyham_path)[:2000]
    all_easyham = [getMsg(easyham_path+file) for file in easyham_dir]
    tdm_easyham = getTdm(all_easyham)
    frequency = tdm_easyham.sum(axis=0)
    density = tdm_easyham.apply(lambda col: col.sum()/frequency.sum(), axis=0)
    occurrence = tdm_easyham.apply(lambda col: (col != 0).sum()/col.count(), axis=0)
    easyham = pd.concat([frequency, density, occurrence], axis=1, join_axes=[frequency.index])
    easyham.columns = ['frequency','density','occurrence']
    easyham['term'] = easyham.index
    easyham = easyham[['term', 'frequency','density','occurrence']]
    easyham = easyham.sort_values(by='occurrence', ascending=False)
    easyham.to_csv("data/easyham.csv", sep=',', encoding='latin1', index=False)

def classifer():
    global hardham_path, spam2_path, easyham2_path
    # classifer
    spam = pd.Series([spamClassifier(spam2_path+file) for file in listdir(spam2_path)])
    hardham = pd.Series([spamClassifier(hardham_path+file) for file in listdir(hardham_path)])
    easyham = pd.Series([spamClassifier(easyham2_path+file) for file in listdir(easyham2_path)])
    # print result
    result = [
        ['easyham', len(easyham[easyham == False])/len(easyham), len(easyham[easyham == True])/len(easyham)],
        ['hardham', len(hardham[hardham == False])/len(hardham), len(hardham[hardham == True])/len(hardham)],
        ['spam', len(spam[spam == False])/len(spam), len(spam[spam == True])/len(spam)]
        ]
    df = pd.DataFrame(result,columns=['type','Normal mail','Spam mail'])
    df = df.set_index('type')
    print(df)

def main():
    global spam_path, easyham_path, hardham_path, spam2_path, easyham2_path
    easyham_path = 'data/easy_ham/'
    easyham2_path = 'data/easy_ham_2/'
    spam_path = 'data/spam/'
    spam2_path = 'data/spam_2/'
    hardham_path = 'data/hard_ham/'
    training()
    classifer()

if __name__ == "__main__":
    main()