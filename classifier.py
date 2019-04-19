import pdb
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble

df = pd.read_csv('New_Observations_SPSS.csv')
df_onehot = pd.get_dummies(df, columns=['Entering Point (8 Direction)', 'Leaving Point(8 Direction)', 'Age (Child/Young/Middle/Old)', 'Hair color(Black/Blonde/Hat)'])


X = df_onehot.values[:, 1:]
Y = df_onehot.values[:, 0].astype(int)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
#clf = sklearn.svm.SVC()
#clf = sklearn.svm.LinearSVC()
#clf = sklearn.ensemble.GradientBoostingClassifier()
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X_train, Y_train)

train_number_of_one = sum(Y_train == 1)
test_number_of_one = sum(Y_test == 1)
print('train one ratio', train_number_of_one/len(Y_train))
print('test one ratio', test_number_of_one/len(Y_test))
print('train accuracy', clf.score(X_train, Y_train))
print('test accuracy', clf.score(X_test, Y_test))
print('OK')
