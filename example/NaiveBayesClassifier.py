from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
import numpy as np


#Multinomial Naive Bayes

# train data
d1 = [2,1,1,0,0,0,0,0,0]
d2 = [1,1,0,1,1,0,0,0,0]
d3 = [0,1,0,0,1,1,0,0,0]
d4 = [0,1,0,0,0,0,1,1,1]
train_data = np.array([d1,d2,d3,d4])
label = np.array(['B','B','B','N'])

# test data
d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

# call multinomialNB
model = MultinomialNB()
# trainning
model.fit(train_data, label)

# test
print('-----MNB-----')
print('Predicting class of d5:', str(model.predict(d5)[0]))
print('Probability of d5 in each class:', model.predict_proba(d5))
print('Predicting class of d6:', str(model.predict(d6)[0]))
print('Probability of d6 in each class:', model.predict_proba(d6))


#Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
# train data
d1 = [1,1,1,0,0,0,0,0,0]
d2 = [1,1,0,1,1,0,0,0,0]
d3 = [0,1,0,0,1,1,0,0,0]
d4 = [0,1,0,0,0,0,1,1,1]
train_data = np.array([d1,d2,d3,d4])
label = np.array(['B','B','B','N']) # 0-B, 1-N

# test data
d5 = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

# call BernoulliNB
model = BernoulliNB()
# training
model.fit(train_data, label)
# test
print('-----BNB-----')
print('Predicting class of d5:', str(model.predict(d5)[0]))
print('Probability of d5 in each class:', model.predict_proba(d5))
print('Predicting class of d6:', str(model.predict(d6)[0]))
print('Probability of d6 in each class:', model.predict_proba(d6))