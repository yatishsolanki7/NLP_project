# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:16:22 2018

@author: sony
"""

#NLP
 #importing libraries
 import numpy as np
 import matplotlib.pyplot as plt
 import pandas as pd
 #importing the daset
 dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t', quoting=3)
 
 #cleaning the texts
 import re
 import nltk 
 nltk.download('stopwords')
 from nltk.corpus import stopwords
 from nltk.stem.porter import PorterStemmer
 corpus=[]
 
 for i in range(0,1000):
     review =re.sub('[^a-zA-z]',' ',dataset['Review'][i])
     review=review.lower()
     review=review.split()
     ps=PorterStemmer()
     review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
     review=' '.join(review)
     corpus.append(review)
     
 #bag of words models
 from sklearn.feature_extraction.text import CountVectorizer
 cv=CountVectorizer()
 X=cv.fit_transform(corpus).toarray()
 y=dataset.iloc[:,1].values
 
 #trainnig the bag of model using classification
 from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 
 
 
 
 
 
 
 
 
 
 
 
 