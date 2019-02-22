import numpy as np
import pandas as pd
import os
import random

import collections

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib
#########################################################3

print('-> Reading source dataset ...')
df = pd.read_csv(os.path.join('ds_neuro.csv'))


print('Columns:',len(df.columns),'| Rows:',len(df))
print('Done')

df.head()

X = df.drop('Output', axis = 1)
y = df['Output']
print('X =',X.shape)
print('y =',y.shape)

Xdata = X.values
Ydata = y.values

scaler = MinMaxScaler()
print(scaler.fit(Xdata))

Xdata = scaler.transform(Xdata)
#print (Xdata)

print (collections.Counter(Ydata))
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(Xdata, Ydata)

print (collections.Counter(y_resampled))
print (len(X_resampled))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y_resampled)

print (collections.Counter(y_train))
print (collections.Counter(y_test))

df_tr_scaler = pd.DataFrame(X_train, columns=X.columns)
df_tr_scaler["Output"]= y_train
print("---> Saving scalled training ...")
df_tr_scaler.to_csv("scalled_train.csv", index=False)
print("Done!")

df_ts_scaler = pd.DataFrame(X_test, columns=X.columns)
df_ts_scaler["Output"]= y_test
print("---> Saving test ...")
df_ts_scaler.to_csv("scalled_test.csv", index=False)
print("Done!")

sample_weight = [120 if i == 1 else 0 for i in y_train]
np.set_printoptions(threshold=np.nan)

f = open("resultados.txt","w+")

########RANDOM FOREST:########
print("********************RESULTS FROM RANDOM FOREST:")
alg =RandomForestClassifier(n_estimators=100)
f.write("Random Forest:\n")

parameters = {'n_estimators':(10, 20, 30, 40, 50, 60, 70, 80, 90, 100), 'criterion':('gini', 'entropy'), 'bootstrap':(True,False)}

alg = GridSearchCV(alg, parameters, cv=5, n_jobs=-1)

alg.fit(X_train,y_train)
s = alg.best_params_
f.write(str(s))
f.write("\n")

print('Training Accuracy = ', alg.score(X_train, y_train))
f.write('Training Accuracy = ')
f.write(str(alg.score(X_train, y_train)))
f.write("\n")
print('Test Accuracy = ', alg.score(X_test, y_test))
f.write('Test Accuracy = ')
f.write(str(alg.score(X_test, y_test)))
f.write("\n")

print(alg.best_estimator_.feature_importances_)
f.write(str(alg.best_estimator_.feature_importances_))

joblib.dump(alg.best_estimator_,"my_randomForest.pkl")

# #########KNN:###########
print("********************RESULTS FROM KNN:")
alg =KNeighborsClassifier()
f.write("KNN:\n")

parameters = {'n_neighbors':(2,5,15,20), 'weights':('uniform', 'distance'),'leaf_size':(10,50,100)}

alg = GridSearchCV(alg, parameters, cv=5, n_jobs=-1)

alg.fit(X_train,y_train)
s = alg.best_params_
f.write(str(s))
f.write("\n")

print('Training Accuracy = ', alg.score(X_train, y_train))
f.write('Training Accuracy = ')
f.write(str(alg.score(X_train, y_train)))
f.write("\n")
print('Test Accuracy = ', alg.score(X_test, y_test))
f.write('Test Accuracy = ')
f.write(str(alg.score(X_test, y_test)))
f.write("\n")

joblib.dump(alg.best_estimator_,"my_kNN.pkl")

#########LOGISTIC REGRESSION:###########
print("********************RESULTS FROM LOGISTIC REGRESSION:")
alg =LogisticRegression(solver='lbfgs',max_iter = 1000)
f.write("Logistic Regression:\n")

parameters = {'fit_intercept':(True,False),'solver':('newton-cg', 'lbfgs','liblinear','sag','saga'),'max_iter':(700,800,900,1000,1100,1200),'multi_class':('ovr','auto'),'warm_start':(True,False)}

alg = GridSearchCV(alg, parameters, cv=5, n_jobs=-1)


alg.fit(X_train,y_train)
s=alg.best_params_
f.write(str(s))
f.write("\n")

print('Training Accuracy = ', alg.score(X_train, y_train))
f.write('Training Accuracy = ')
f.write(str(alg.score(X_train, y_train)))
f.write("\n")
print('Test Accuracy = ', alg.score(X_test, y_test))
f.write('Test Accuracy = ')
f.write(str(alg.score(X_test, y_test)))
f.write("\n")


#########SVC:###########
print("********************RESULTS FROM SVC:")
alg = SVC()
f.write("SVC:\n")

parameters = { 'degree':(2,3,4), 'gamma':('auto','scale'),'shrinking':(True,False),'probability':(True,False),'decision_function_shape':('ovo','ovr')} #'p':(1,2,3),

alg = GridSearchCV(alg, parameters, cv=5, n_jobs=-1)

alg.fit(X_train_transf, y_train) 
s=alg.best_params_
f.write(str(s))
f.write("\n")

print('Training Accuracy = ', alg.score(X_train, y_train))
f.write('Training Accuracy = ')
f.write(str(alg.score(X_train, y_train)))
f.write("\n")
print('Test Accuracy = ', alg.score(X_test, y_test))
f.write('Test Accuracy = ')
f.write(str(alg.score(X_test, y_test)))
f.write("\n")


#########PERCEPTRON MULTICAPA:###########
print("********************RESULTS FROM MLP:")
alg = MLPClassifier()
f.write("MLPC:\n")

parameters = {'hidden_layer_sizes':((5,1),(5,2),(5,3),(5,4),(5,5),(5,10)),'activation':('identity','logistic','tanh','relu'), 'solver':('lbfgs','sgd','adam')}

alg = GridSearchCV(alg, parameters, cv=5, n_jobs=-1)

alg.fit(X_train, y_train) 
s=alg.best_params_
f.write(str(s))
f.write("\n")

print('Training Accuracy = ', alg.score(X_train, y_train))
f.write('Training Accuracy = ')
f.write(str(alg.score(X_train, y_train)))
f.write("\n")
print('Test Accuracy = ', alg.score(X_test, y_test))
f.write('Test Accuracy = ')
f.write(str(alg.score(X_test, y_test)))
f.write("\n")

f.close()
