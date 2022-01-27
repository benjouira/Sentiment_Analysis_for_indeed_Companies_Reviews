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
