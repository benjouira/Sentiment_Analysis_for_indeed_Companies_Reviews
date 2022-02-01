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


# 3. Sentiment Analysis

# NLTK already has a built-in sentiment analyzer model called 'VADER'
vader = SentimentIntensityAnalyzer()

# calculate the scores,for example, of the 10 first reviews
df["new_review"][:10].apply(lambda x: vader.polarity_scores(x)).values

# *****************************************************************
# SentimentIntensityAnalyzer() is an object and polarity_scores is a method which will  give us scores of the following categories:
#     Positive
#     Negative
#     Neutral
#     Compound

# The compound score is the sum of positive, negative & neutral scores which is then normalized between -1(most extreme negative) and +1 (most extreme positive).

# The more Compound score closer to +1, the higher the positivity of the text.
# ****************************************


# let's add the compound score foreach reviews
df["polarity_score"] = df["new_review"].apply(lambda x: vader.polarity_scores(x)["compound"])
df.head(2)


# 4. Sentiment Modeling

# Feature Engineering

# creating Target from polarity
df["polarity_label"] = df["polarity_score"].apply(lambda x: "pos" if x > 0.6 else "neg" if x < 0 else "neu")
df.head(2)

# let's check if we have unbalanced data problem let's look at a target value count
sns.countplot(df['polarity_label'])

# encoding the new target
df["polarity_label"] = LabelEncoder().fit_transform(df["polarity_label"])
x = df["new_review"]
y_polarity = df["polarity_label"]
y_raiting = df["rating"]

# TF-IDF

# word tf-idf
tf_idf_vectorizer = TfidfVectorizer()
x_tf_idf = tf_idf_vectorizer.fit_transform(x)


# 5. Modeling

# Logistic Regression

# training my model with polarity target
model_with_polarity = LogisticRegression().fit(x_tf_idf, y_polarity)

# for better result runinig my model with 5 different fold  
cross_val_score(model_with_polarity, x_tf_idf, y_polarity, scoring="accuracy", cv=5).mean()


# evaluate my model 

test1 = pd.Series("this product is great")
test2 = pd.Series("look at that shit very bad")
test3 = pd.Series("it's normal")

res_test1 = CountVectorizer().fit(x).transform(test1)
res_test2 = CountVectorizer().fit(x).transform(test2)
res_test3 = CountVectorizer().fit(x).transform(test3)

print("sentence 1 : ",model_with_polarity.predict(res_test1))
print("sentence 2 : ",model_with_polarity.predict(res_test2))
print("sentence 3 : ",model_with_polarity.predict(res_test3))



# training my model with rating target
model_with_rating = LogisticRegression().fit(x_tf_idf, y_raiting)

# for better result runinig my model with 5 different fold  
cross_val_score(model_with_rating, x_tf_idf, y_raiting, scoring="accuracy", cv=5).mean()


