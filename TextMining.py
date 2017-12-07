import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



dataset = pd.read_csv("Restaurant_Reviews.tsv", sep = "\t")

#select only letters
review = re.sub('[^a-zA-Z]', ' ', dataset["Review"][0])

#convert letters to lower case
review = review.lower()

#plit the sent to words
review = review.split()

ps = PorterStemmer()

#review = [word for word in review if word not in set(stopwords.words('english'))]

#remove stop words and stem
#stemming - considering only root of the word not different versions of the same word - to avoid too much of sparsity

review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]











