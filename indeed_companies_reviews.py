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






