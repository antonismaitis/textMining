__author__ = "Team Antonis - Giwrgos"
__version__ = "1.0"
__email__ = "amaitisn@csd.auth.gr"

"""
Code of HomeWork in Text mining course
of the MSc Course on the subject of Machine Learning for the Aristotle University of Thessaloniki.
"""


from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import graphviz
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/anton/PycharmProjects/textMinning_Project/venv/Lib/site-packages/graphviz-2.38/release/bin/'
# path to graphiz antonis PC - you can use yours by adding mine in comments
# try 1 - 5 - 7 . What changes and why?
seed = 7

# load the dataset from csv file
with open('C:/Users/anton/Desktop/MSc/MachineLearning/databases/IRIS.csv') as f:
	csvreader = csv.reader(f, delimiter=',', quotechar='|')
	iris_dataset = list(csvreader)
	print("IRIS dataset loaded.")

print("Example contents of data loaded:")
for i in range(0,10):
	print("{}".format(iris_dataset[i]))

# X / Y split
print("Spliting data & labels")
X = np.array(iris_dataset)[:,0:7]
Y = np.array(iris_dataset)[:,8]
print(X)
# Splitting the dataset to training and testing instances
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
validation_size = 0.20
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Training and Testing
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
clf = DecisionTreeClassifier(random_state=seed)
clf.fit(X_train,Y_train)
predictions = clf.predict(X_test)

# Result evaluation
acc = metrics.accuracy_score(Y_test,predictions)
prec = metrics.precision_score(Y_test,predictions,average='micro')
rec = metrics.recall_score(Y_test,predictions,average='micro')
f1 = metrics.f1_score(Y_test,predictions,average='micro')
print('Results: \n')
print('Accracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))



