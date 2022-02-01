from time import time
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import spacy
import re
import nltk
from spacy.lang.fr.stop_words import STOP_WORDS
from string import punctuation
import seaborn as sns
from matplotlib import style
style.use("fivethirtyeight")

# ******************

# Reading the dataset 
df = pd.read_csv('indeed_reviews.csv' ,delimiter='~')
df.head(2)

# *******************
df=df[['title','review','employee_position','employee_state','city','state','date_review','rating']]
df.head(2)

# ******************
df.info()

# ***************
sns.countplot(df['rating'])

# *****************
# *1. Text Preprocessing*
# Normalizing Case Folding
df['review'] = df['review'].str.lower()

# Punctuations
df['review'] = df['review'].str.replace('[^\w\s]', '')

# Numbers
df['review'] = df['review'].str.replace('\d', '')


# spelling correction
def orth_corr(text):
    try:
        return TextBlob(text).correct()
    except:
        return None

# Stopwords remove
sw = stopwords.words('english')
df['new_review'] = df['new_review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# detecting Rare Words
drops = pd.Series(' '.join(df['new_review']).split()).value_counts() 
drops = drops [drops < 3]
drops

# delete Rare Words
df['new_review'] = df['new_review'].apply(lambda x: " ".join(y for y in x.split() if y not in drops))

# Lemmatization

lemmatizer = WordNetLemmatizer()
df['new_review'] = df['new_review'].apply(lambda x: " ".join([word.lemma_ for word in nlp(x)]))


# *2. Text Visualization*

# Term Frequencies

tf = pd.DataFrame(' '.join(df['new_review']).split()).value_counts() 

tf.describe()



# Wordcloud

text = " ".join(i for i in df.new_review)
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



