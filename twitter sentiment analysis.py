#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Step-1: Import the Necessary Dependencies


# In[4]:


#NECCESSARY DEPENDENCY PACKAGE

import re
import numpy as np
import pandas as pd
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


#Step-2: Read and Load the Dataset


# In[10]:


# Importing the dataset
DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('C:/Users/ADMIN/Desktop/twitter.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
df.sample(5)


# In[ ]:


#Step-3: Exploratory Data Analysis


# In[ ]:


# Five top records of data


# In[11]:


df.head()


# In[ ]:


#Columns/features in data


# In[12]:


df.columns


# In[ ]:


#Length of the dataset


# In[13]:


print('length of data is', len(df))


# In[ ]:


#Shape of data


# In[14]:


df. shape


# In[ ]:


#Data information


# In[15]:


df.info()


# In[ ]:


#Datatypes of all columns


# In[16]:


df.dtypes


# In[ ]:


#Checking for null values


# In[17]:


np.sum(df.isnull().any(axis=1))


# In[ ]:


#Rows and columns in the dataset


# In[18]:


print('Count of columns in the data is:  ', len(df.columns))
print('Count of rows in the data is:  ', len(df))


# In[ ]:


#Check unique target values


# In[19]:


df['target'].unique()


# In[ ]:


#Check the number of target values


# In[20]:


df['target'].nunique()


# In[ ]:


#Data Visualization of Target Variables


# In[ ]:


# Plotting the distribution for dataset.
ax = df.groupby('target').count().plot(kind='bar', title='Distribution of data',legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)
# Storing data in lists.
text, sentiment = list(df['text']), list(df['target'])


# In[ ]:


#Data Preprocessing


# In[22]:


import seaborn as sns
sns.countplot(x='target', data=df)


# In[ ]:


#Selecting the text and Target column for our further analysis


# In[24]:


data=df[['text','target']]


# In[ ]:


#Replacing the values to ease understanding. (Assigning 1 to Positive sentiment 4)


# In[25]:


data['target'].unique()


# In[ ]:


#Printing unique values of target variables


# In[26]:


data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]


# In[ ]:


#Separating positive and negative tweets


# In[27]:


data_pos = data_pos.iloc[:int(20000)]
data_neg = data_neg.iloc[:int(20000)]


# In[ ]:


#Taking one-fourth of the data so we can run it on our machine easily


# In[28]:


dataset = pd.concat([data_pos, data_neg])


# In[ ]:


#Making statement text in lowercase


# In[29]:


dataset['text']=dataset['text'].str.lower()
dataset['text'].tail()


# In[ ]:


#Defining set containing all stopwords in English.


# In[30]:


stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


# In[ ]:


#Cleaning and removing the above stop words list from the tweet text


# In[ ]:


TOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text))
dataset['text'].head()


# In[ ]:


#Cleaning and removing punctuations


# In[32]:


import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
dataset['text']= dataset['text'].apply(lambda x: cleaning_punctuations(x))
dataset['text'].tail()


# In[ ]:


#Cleaning and removing repeating characters


# In[33]:


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))
dataset['text'].tail()


# In[ ]:


#Cleaning and removing URLs


# In[34]:


def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))
dataset['text'].tail()


# In[ ]:


#Cleaning and removing numeric numbers


# In[35]:


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
dataset['text'].tail()


# In[ ]:


#Getting tokenization of tweet text


# In[36]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'w+')
dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
dataset['text'].head()


# In[ ]:


# Applying stemming


# In[37]:


import nltk
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
dataset['text']= dataset['text'].apply(lambda x: stemming_on_text(x))
dataset['text'].head()


# In[ ]:


#Applying lemmatizer


# In[ ]:


lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))
dataset['text'].head()


# In[ ]:


# Separating input feature and label


# In[39]:


X=data.text
y=data.target


# In[ ]:


#Plot a cloud of words for negative tweets


# In[42]:


data_neg = data['text'][:800000]
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(wc)


# In[ ]:


#Plot a cloud of words for positive tweets


# In[43]:


data_pos = data['text'][800000:]
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
              collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (20,20))
plt.imshow(wc)


# In[ ]:


#Splitting Our Data Into Train and Test Subsets


# In[44]:


# Separating the 95% data for training data and 5% for testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)


# In[ ]:


#Transforming the Dataset Using TF-IDF Vectorizer
#7.1: Fit the TF-IDF Vectorizer


# In[45]:


vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))


# In[46]:


#Transform the data using TF-IDF Vectorizer

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


# In[ ]:


#Function for Model Evaluation
def model_Evaluate(model):
# Predict values for Test dataset
y_pred = model.predict(X_test)
# Print the evaluation metrics for the dataset.
print(classification_report(y_test, y_pred))
# Compute and plot the Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
categories = ['Negative','Positive']
group_names = ['True Neg','False Pos', 'False Neg','True Pos']
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
xticklabels = categories, yticklabels = categories)
plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


# In[49]:


#Model Building
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)


# In[51]:


#Plot the ROC-AUC Curve for model-1
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# In[52]:


#Model-2:

SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)


# In[ ]:


#Plot the ROC-AUC Curve for model-2

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#Model-3

LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)


# In[ ]:


#Plot the ROC-AUC Curve for model-3

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




