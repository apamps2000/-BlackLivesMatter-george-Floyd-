# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:14:54 2020

@author: Olatunji Apampa
"""


#############CLEANSING & ANALYSIS BLM2020 ALL COMBINED FINAL2
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
import matplotlib as plt
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("C:/Users/apamps/Documents/DATA ANALYSIS - PYTHON/BlackLivesMatter2020.csv")
data.head()
data.describe()
data.info()

#####Removing Columns with missing data
del data["Unnamed: 0"]
del data["user_lang"]
del data["source"]
del data["profile_image_url"]
del data["status_url"]
data.head()
data.describe()
data.info()

#### Save to a new csv file
data.to_csv("C:/Users/apamps/Documents/DATA ANALYSIS - PYTHON/BlackLivesMatter2020-D.csv", index=False)

####### Load the new dataset
data = pd.read_csv("C:/Users/apamps/Documents/DATA ANALYSIS - PYTHON/BlackLivesMatter2020-D.csv")
data.info()

####Removing Rows with null values
df = data.dropna(how='all')
print(data)

###Data Cleansing
string.punctuation

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Tweet_punct'] = df['text'].apply(lambda x: remove_punct(x))
df.head(10)

###Tokenisation
def tokenization(text):
    text = re.split('\W+', text)
    return text
df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))
df.head()

####Removing Stopwords
##Identifying other word to be removed
(['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
        'treatment', 'associated', 'patients', 'may','day', 'case','old', 'https', 'co'])
 
stopword = nltk.corpus.stopwords.words('english')
             
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
df.head(10)

######Stemming and Lammitization
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text
df['Tweet_stemmed'] = df['Tweet_nonstop'].apply(lambda x: stemming(x))
df.head()

wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text
df['Tweet_lemmatized'] = df['Tweet_nonstop'].apply(lambda x: lemmatizer(x))
df.head()

####Cleaning the text in the tweets
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

####Vectorisation
countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(df['text'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()

#####Sentiment Analysis
from textblob import TextBlob, Blobber 
#### Extracting only the polarity from the SA for the first 2000 tweets
SA = df['text'][:67792].apply(lambda x: TextBlob(x).sentiment.polarity)
print(SA)

df['sentiment'] = SA
print(df)

pos_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['sentiment'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['sentiment'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['sentiment'][index] < 0]

####We print percentages
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(df['text'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(df['text'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(df['text'])))

#####Word Cloud
###removing punctuations
import matplotlib.cm as cm
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
all_doc = df['text'].str.split(' ')
all_doc.head()

###Joining all of the texts in the tweets
all_doc_cleaned = []

for text in all_doc:
    text = [x.strip(string.punctuation) for x in text]
    all_doc_cleaned.append(text)

all_doc_cleaned[0]

text_doc = [" ".join(text) for text in all_doc_cleaned]
final_text_doc = " ".join(text_doc)
final_text_doc[:500]

stopwords = set(STOPWORDS)
stopwords.update(["subject","re","vince","kaminski","enron","cc", "will", "s", "1","e","t", "https", "co", "com"])

###WordCloud
import matplotlib.pyplot as plt 
pd.set_option('display.max_colwidth', 100)
wordcloud_doc = WordCloud(background_color="black").generate(final_text_doc)

plt.figure(figsize = (20,20))
plt.imshow(wordcloud_doc, interpolation='bilinear')
plt.axis("off")
plt.show() 

####the most popular words as a frequency table
import collections 
filtered_words_doc = [word for word in final_text_doc.split() if word not in stopwords]
counted_words_doc = collections.Counter(filtered_words_doc)

word_count_doc = {}

for letter, count in counted_words_doc.most_common(30):
    word_count_doc[letter] = count
    
for i,j in word_count_doc.items():
        print('Word: {0}, count: {1}'.format(i,j))
        
        
########Tweets by Country

        









