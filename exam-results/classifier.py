import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

# This is for Support Vector Machines 
def svm_clf (x_train, y_train):
   print("SVM Classifier") 
   #clf = svm.LinearSVC()
   #clf = svm.SVC(kernel='rbf')
   clf = svm.SVC(kernel='poly', degree=8)
   clf.fit(x_train, y_train)
   return clf 


# This is for Logistic Regression 
def logistic_clf (x_train, x_test):
    print("Logistic  Classifier") 
    clf = LogisticRegression(C=1e5)
    clf.fit(x_train, y_train)
    return clf


# This is for Naive Bayes 
def naive_bayes (x_train, x_test):
   print("Naive Bayes  Classifier") 
   clf = GaussianNB()
   clf.fit(x_train, y_train)
   return clf


# This is for Decision Tree  
def tree_clf (x_train, y_train):
    print("Decision Tree  Classifier") 
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    return clf

# This is knn classifier 

def knn_clf (x_train, y_train):
    print("KNN  Classifier") 
    n_neighbors = 15
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(x_train, y_train)
    return clf 



if __name__ == "__main__":

   df = pd.read_csv(sys.argv[1])

   print("Input data frame:", df.shape)
   print("Columns:", df.columns)

   data = df.to_numpy()

   X = data[:,1:7]
  
   # for binary set 
   #Y = data[:,8]

   # for multi-class
   Y = data[:,9]
   
   x_train, x_test, y_train, y_test \
      = train_test_split(X, Y, test_size=0.20, random_state=42)

   models = [svm_clf, logistic_clf, naive_bayes, tree_clf, knn_clf]

   conf_mat = [] 
   for m in models:
      clf = m(x_train, y_train)    
      y_hat = clf.predict(x_test)
      cm = confusion_matrix(y_test, y_hat)
      print("Confusion matrix:\n", cm)
      print("========")
