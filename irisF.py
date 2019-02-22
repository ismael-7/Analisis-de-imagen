import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import preprocessing

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from scipy import stats

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

##############
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
##################
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def check_NaN(df):
	if (df.isnull().values.any() == False):
		print("There are no NaN values in the dataset! Proceed!")
	else:
		response = input("NaNs found in dataset. Proceed to delete NaN values? y/n ")
		if response == 'n':
			response2 = input("User choose to ignore NaN values! Do you want to exit? y/n ")
			if response == 'n':
				print("Proceding with NaNs in dataset!")
			elif response == 'y':
				exit()
			else:
				print("Unknown input detected. Aborting program!")
				exit()
		elif response == 'y':
			print("Deleting NaNs")
			df.dropna()
		else:
			print("Unknown input detected. Aborting program!")
			exit()
	return(df)

def delete_outliers(X,Y,threshold):
	is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
	pos_strings = np.where(is_number(X.dtypes) == False)[0]
	
	aux = X
	for i in pos_strings:
		aux = aux.drop(X.columns[i],1)
	
	k2, p = stats.normaltest(aux)

	if p.any() < 0.05:
		aux = preprocessing.normalize(aux, norm='l2')

	outliers = np.where((np.abs(stats.zscore(aux)) > threshold).all(axis=1))[0]
	
	print("The number of outliers is: ", len(outliers))

	for ii in outliers:
		X.drop(X.index[ii],inplace=True)
		Y.drop(Y.index[ii],inplace=True)

	return (X,Y)
	
def delete_outliers2(df,outliers):

	for i in outliers:
		if i == -1:
			df.drop(df.index[i],inplace=True)

	return df

iris = sns.load_dataset("iris")

#Check and Delete NaNs
iris = check_NaN(iris)

#Delete colums with constant values:
iris.loc[:, (iris != iris.iloc[0]).any()]

print("Shape of data:")
print(iris.shape)
print("Head of data:")
print(iris.head(10))
print("Description of Data:")
print(iris.describe())

sns.set(style="ticks")
piris = pd.melt(iris, "species", var_name="measurement") 
sns.boxplot(x="measurement",y="value",hue="species",palette="bright",data=piris)
sns.despine(offset=10,trim=True)
plt.savefig('Imagenes/iris3.png')
plt.show() 
sns.set(style="ticks")
sns.pairplot(iris,hue="species",palette="bright")
plt.savefig('Imagenes/iris1.png')
plt.show()
sns.catplot(x="measurement", y="value", hue="species", data=piris, height=7, kind="bar",palette="bright") 
plt.savefig('Imagenes/iris2.png')
plt.show() 

####################################
y = iris.species
X = iris.drop('species',axis=1)
#Delete outliers
X,y = delete_outliers(X,y,4)

#scaler = MinMaxScaler()
#scaler.fit(X)
#X = scaler.transform(X)

#Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42,
                                                    stratify=y)



#Define models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

#for train_index, test_index in kfold.split(X,y):
	#X_train,X_test=X.iloc[train_index],X.iloc[test_index]
	#y_train,y_test=y.iloc[train_index],y.iloc[test_index]
	
	#Splitvalues.append((X_train,y_train,X_test,y_test))
	
#for name, model in models:
	#for trainX,trainy,testX,testy in Splitvalues:
		
		#model.fit(trainX,trainy)
		#resultsIt.append((model.score(trainX,trainy),model.score(testX,testy))
		
	#resultsF.append((name,resultsIt))

kf = KFold(n_splits=10, shuffle = True, random_state=42,)
print("Número de splits",kf.get_n_splits(X))

dic={'LR':{"TrainingScore":[],"TestScore":[],"Accuracy":[],"PRFS":[]},'LDA':{"TrainingScore":[],"TestScore":[],"Accuracy":[],"PRFS":[]},
     'QDA':{"TrainingScore":[],"TestScore":[],"Accuracy":[],"PRFS":[]},'KNN':{"TrainingScore":[],"TestScore":[],"Accuracy":[],"PRFS":[]},
     'CART':{"TrainingScore":[],"TestScore":[],"Accuracy":[],"PRFS":[]},'SVM':{"TrainingScore":[],"TestScore":[],"Accuracy":[],"PRFS":[]}}

resul_train=[]
resul_test=[]

for train_index,test_index in kf.split(X,y):

	X_train,X_test=X.iloc[train_index],X.iloc[test_index]
	y_train,y_test=y.iloc[train_index],y.iloc[test_index]
	
	for name, model in models:
		model.fit(X_train,y_train)
		y_pred = model.predict(X_test)
		
		res_PRFS = precision_recall_fscore_support(y_test,y_pred)
		res_acc = accuracy_score(y_test,y_pred)
		
		res_train=model.score(X_train, y_train)
		res_test=model.score(X_test, y_test)
		
		dic[name]["TrainingScore"].append(res_train)
		dic[name]["TestScore"].append(res_test)
		dic[name]["Accuracy"].append(res_acc)
		dic[name]["PRFS"].append(res_PRFS)

#Ver medidas antes de comparar la precision
# for i in dic:
# 	print('modelo:',i,'medidas',dic[i]['PRFS'])
# exit()
print(stats.kruskal(dic['LDA']["Accuracy"], dic['LR']["Accuracy"]))

#exit()
#for name, model in models:
	#cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	#cv_resultsTest = cross_val_score(model, X_test, y_test, cv=kfold, scoring='accuracy')
	
	#resultsTr.append(cv_results)
	#resultsTe.append(cv_resultsTest)
	
	#names.append(name)
	
	#msgTr = "Training Mean Score: %s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
	#msgTe = "Test Mean Score: %s: %f (%f)\n" % (name, cv_resultsTest.mean(), cv_resultsTest.std())
	
	#f.write(msgTr)
	#f.write(msgTe)
	#f.write("***************************************\n")

## Compare Algorithms
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(resultsTr)
#ax.set_xticklabels(names)
#plt.savefig('Imagenes/algorithms.png')
#plt.show()

