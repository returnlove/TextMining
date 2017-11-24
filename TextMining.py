import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

dataset = pd.read_csv("Restaurant_Reviews.tsv", sep = "\t")


#select only letters

review = re.sub('[^a-zA-Z]', ' ', dataset["Review"][0])