import pandas as pd
import string
import email
import nltk
import os
import pickle
import re
from nltk.tokenize import RegexpTokenizer
#import sklearn.cross_validation as skcv
import warnings
warnings.filterwarnings('ignore')
import sklearn.feature_extraction.text as skft
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skmetrics
import sklearn.pipeline as skpipe
import sklearn.decomposition as skd
import sklearn.naive_bayes as sknb
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import wordcloud
from sklearn.model_selection import cross_val_score, cross_val_predict
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
import numpy as np

def create_phish_df(my_dir):
    titles = []
    contents = []
    labels = []

    for f in os.listdir(os.path.join('phish',my_dir)):
            with open(os.path.join('phish', my_dir, f), 'r') as reader:
                try:
                    c = reader.read()
                except:
                    continue
                contents.append(c)
                titles.append(f)
                labels.append('phish')

    df = pd.DataFrame({'title': titles, 'content': contents, 'label': 'phish'},
                        columns = ['label', 'title', 'content'])
    return df

phish_email_list = [r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\phish\20051114", r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\phish\phishing0", r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\phish\phishing1", r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\phish\phishing2", r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\phish\phishing3"]

phish_lst = []
for phish_folder in phish_email_list:
    phish_lst.append(create_phish_df(phish_folder))

df_phish = pd.concat(phish_lst)
df_phish = df_phish[:5000]

ham_email_list = [r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\enron3", r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\enron4", r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\enron5", r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\enron6"]

def create_ham_df(my_dir):
    titles = []
    contents = []
    labels = []

    for f in os.listdir(os.path.join(my_dir,'ham')):
            with open(os.path.join(my_dir, 'ham', f), 'r') as reader:
                try:
                    c = reader.read()
                except:
                    continue
                contents.append(c)
                titles.append(f)
                labels.append('ham')

    df = pd.DataFrame({'title': titles, 'content': contents, 'label': 'ham'},
                        columns = ['label', 'title', 'content'])
    return df

ham_list = []
for ham in ham_email_list:
    ham_list.append(create_ham_df(ham))

df_ham = pd.concat(ham_list)
df_ham = df_ham[:5000]

df_emails = pd.concat([df_ham, df_phish])

def contains_phish_link(df):
    #match = re.match(r'True', str(df.title))
    if(re.match(r'.*True.txt$', str(df.title))):
        return True
    return False

df_emails['malic'] = df_emails.apply(contains_phish_link, axis = 1)

df_emails_train, df_emails_test = train_test_split(df_emails, test_size=0.3, random_state=0)

#frequency distribution
text_all = '\n'.join(df_emails_train.content).lower()
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')#nltk.tokenize.wordpunct_tokenize(text_all)
tokens_all = tokenizer.tokenize(text_all)
tokens_all = [word for word in tokens_all if word not in stop_words and word != 'font' and word != 'subject']#word not in string.punctuation

fd = nltk.probability.FreqDist(tokens_all)

phish_text_all = '\n'.join(df_phish.content).lower()
phish_tokens_all = tokenizer.tokenize(phish_text_all)
phish_tokens_all = [word for word in phish_tokens_all if word not in stop_words and word != 'font' and word != 'subject']

fd_phish = nltk.probability.FreqDist(phish_tokens_all)

def netcraft_algo(df):
    if(df['malic'] == True):
        return 'phish'
    else:
        pass

def netcraft_cv(df, model):
    precision_scores_weighted = []
    precision_scores_micro = []
    precision_scores_macro = []
    
    precision_scores = []
    recall_scores = []
    
    f1_scores_macro = []
    f1_scores_micro = []
    f1_scores_weighted = []
    
    recall_scores_macro = []
    recall_scores_micro = []
    recall_scores_weighted = []
    
    
    accuracy_scores = []
    kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
    fold = 0
    for train_ind, test_ind in kf.split(df):

        df_emails_train = df.iloc[train_ind]
        df_emails_test =  df.iloc[test_ind]
        
        model.fit(df_emails_train.content, df_emails_train.label)
        model_test_predicted = model.predict(df_emails_test.content)

        df_emails_test['predicted_label'] = df_emails_test.apply(netcraft_algo, axis = 1)
        netcraft_test_predicted = [x for x in df_emails_test['predicted_label']]
        test_predicted = []
        for a, b in zip(netcraft_test_predicted, model_test_predicted):
            if((a == 'phish') and (b != 'phish')):
                test_predicted.append(a)
            else:
                test_predicted.append(b)
        print('fold', fold)
        print (skmetrics.classification_report(df_emails_test.label, test_predicted))
        accuracy_scores.append(skmetrics.accuracy_score(df_emails_test.label, test_predicted))
        
        f1_scores_macro.append(skmetrics.f1_score(df_emails_test.label, test_predicted, average = 'macro'))
        f1_scores_micro.append(skmetrics.f1_score(df_emails_test.label, test_predicted, average = 'micro'))
        f1_scores_weighted.append(skmetrics.f1_score(df_emails_test.label, test_predicted, average = 'weighted'))
        
        precision_scores.append(skmetrics.precision_score(df_emails_test.label, test_predicted, pos_label = 'phish'))
        
        precision_scores_macro.append(skmetrics.precision_score(df_emails_test.label, test_predicted, average = 'macro'))
        precision_scores_micro.append(skmetrics.precision_score(df_emails_test.label, test_predicted, average = 'micro'))
        precision_scores_weighted.append(skmetrics.precision_score(df_emails_test.label, test_predicted, average = 'weighted'))
        
        recall_scores.append(skmetrics.recall_score(df_emails_test.label, test_predicted, pos_label = 'phish'))
        
        recall_scores_macro.append(skmetrics.recall_score(df_emails_test.label, test_predicted, average = 'macro'))
        recall_scores_micro.append(skmetrics.recall_score(df_emails_test.label, test_predicted, average = 'micro'))
        recall_scores_weighted.append(skmetrics.recall_score(df_emails_test.label, test_predicted, average = 'weighted'))
        
        
        fold += 1
    
    print('F1 Micro:', np.mean(f1_scores_micro))
    print('F1 Macro:', np.mean(f1_scores_macro))
    print('F1 Weighted:', np.mean(f1_scores_weighted))
    print()
    print('Precision Micro:', np.mean(precision_scores_micro))
    print('Precision Macro:', np.mean(precision_scores_macro))
    print('Preicision Weighted:', np.mean(precision_scores_weighted))
    print()
    print('Recall Micro:', np.mean(recall_scores_micro))
    print('Recall Macro:', np.mean(recall_scores_macro))
    print('Recall Weighted:', np.mean(recall_scores_weighted))
    print()
    print('avg accuracy:', round(np.mean(accuracy_scores), 2))
    print('avg phish precision:', round(np.mean(precision_scores), 2))
    print('avg phish recall:', round(np.mean(recall_scores), 2))

pipeline = skpipe.Pipeline(
    steps = [('vect', skft.CountVectorizer(max_df=0.7)),
     ('tfidf', skft.TfidfTransformer()),
     ('clf', sknb.MultinomialNB())])

df_emails_train, df_emails_test = train_test_split(df_emails, test_size=0.3, random_state=0)
pipeline.fit(df_emails_train.content, df_emails_train.label)

nb_test_predicted = pipeline.predict(df_emails_test.content)

df_emails_test['predicted_label'] = df_emails_test.apply(netcraft_algo, axis = 1)

netcraft_test_predicted = [x for x in df_emails_test['predicted_label']]

test_predicted = []
for a, b in zip(netcraft_test_predicted, nb_test_predicted):
    #print(a, b)
    if((a == 'phish') and (b != 'phish')):
        test_predicted.append(a)
    else:
        test_predicted.append(b)

df_emails_test['predicted_label'] = test_predicted

print ('Accuracy:', skmetrics.accuracy_score(df_emails_test.label, test_predicted))

kfold = KFold(n_splits=2, random_state=0, shuffle = True)

print (skmetrics.classification_report(df_emails_test.label, test_predicted))

nb_scores = cross_val_predict(pipeline, df_emails.content, df_emails.label, cv=kfold)

print(skmetrics.classification_report(df_emails.label, nb_scores))

netcraft_cv(df_emails, pipeline)

titles = []
contents = []
labels = []

for f in os.listdir(os.path.join(r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\enron6",'spam')):
        with open(os.path.join(r"C:\Users\Sreelekshmi\Downloads\Phishing-Detection-master\Phishing-Detection-master\enron6", 'spam', f), 'r') as reader:
            try:
                c = reader.read()
            except:
                continue
            contents.append(c)
            titles.append(f)
            labels.append('ham')

df_spam = pd.DataFrame({'title': titles, 'content': contents, 'label': 'spam'},
                    columns = ['label', 'title', 'content'])

predictions = pipeline.predict(df_spam.content)

df_spam['predicted_label'] = predictions

