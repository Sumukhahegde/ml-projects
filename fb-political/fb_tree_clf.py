import sys
import numpy as np 
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

if __name__ == "__main__":

    df = pd.read_csv(sys.argv[1])
    df.drop(df.filter(regex="Unname"),axis=1, inplace=True)

    print("Input data frame:", df.shape, df.columns)

    df_train, df_test = train_test_split(df, test_size =0.15, random_state=751)

    X_train=df_train.iloc[:,1:6].values
    y_train=df_train.iloc[:,6].values

    X_test = df_test.iloc[:,1:6].values
    y_test = df_test.iloc[:,6].values

    ids = df_test['id'].tolist() 

    #clf = tree.DecisionTreeClassifier(criterion='gini') 
    clf = tree.DecisionTreeClassifier(criterion='entropy') 
    clf.fit(X_train, y_train)

    z = clf.score(X_train, y_train)
    print("score:", z)

    y_hat = clf.predict(X_test)
  
    for i in range (len(y_test)):
        if y_test[i] != y_hat[i]:
           print("Failed:", i, ids[i], "y_true:", y_test[i], "y_hat:", y_hat[i])
        else: 
           print("Passed:", i, ids[i], "y_true:", y_test[i], "y_hat:", y_hat[i])


    f = plt.figure(figsize=(12,8))
    plot_tree(clf, filled=True, fontsize=8)
    #plt.show()
    f.savefig("fb_entropy.pdf", bbox_inches='tight')
             
    print(clf.get_params())
    print(clf.get_n_leaves())
