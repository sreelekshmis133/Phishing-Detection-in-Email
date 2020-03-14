import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

raw_data = pd.read_csv("datasetfinal.csv") 
raw_data['URL'].str.split("://").head()
seperation_of_protocol = raw_data['URL'].str.split("://",expand = True) #expand argument in the split method will give you a new column
seperation_domain_name = seperation_of_protocol[1].str.split("/",1,expand = True)
seperation_domain_name.columns=["domain_name","address"]
splitted_data = pd.concat([seperation_of_protocol[0],seperation_domain_name],axis=1)
splitted_data.columns = ['protocol','domain_name','address']
splitted_data['is_phished'] = pd.Series(raw_data['Target'], index=splitted_data.index)

def long_url(l):
    l= str(l)
    """This function is defined in order to differntiate website based on the length of the URL"""
    if len(l) < 54:
        return 0
    elif len(l) >= 54 and len(l) <= 75:
        return 2
    return 1
	
splitted_data['long_url'] = raw_data['URL'].apply(long_url) 
def have_at_symbol(l):
    """This function is used to check whether the URL contains @ symbol or not"""
    if "@" in str(l):
        return 1
    return 0
	
splitted_data['having_@_symbol'] = raw_data['URL'].apply(have_at_symbol)
def redirection(l):
    """If the url has symbol(//) after protocol then such URL is to be classified as phishing """
    if "//" in str(l):
        return 1
    return 0
	
splitted_data['redirection_//_symbol'] = seperation_of_protocol[1].apply(redirection)
def prefix_suffix_seperation(l):
    if '-' in str(l):
        return 1
    return 0
splitted_data['prefix_suffix_seperation'] = seperation_domain_name['domain_name'].apply(prefix_suffix_seperation)
def sub_domains(l):
    l= str(l)
    if l.count('.') < 3:
        return 0
    elif l.count('.') == 3:
        return 2
    return 1

splitted_data['sub_domains'] = splitted_data['domain_name'].apply(sub_domains)
features = ['long_url', 'having_@_symbol', 'redirection_//_symbol','prefix_suffix_seperation','sub_domains']
X = splitted_data[features]
y = splitted_data.is_phished
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100,n_jobs=2,random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(X, y)
pred=clf.predict(X_test)
list(pred)
print(pred)
clf.predict_proba(X_test)[0:10]
y_test1=y_test.as_matrix()
results = confusion_matrix(y_test1, pred) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(y_test1, pred))
print ('Report : ')
print(classification_report(y_test1, pred))
print(list(zip(X, clf.feature_importances_)))