from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

##########################################################################################################
##################### Importing the dataset
from sklearn import datasets
from sklearn.decomposition import PCA

#import data
iris = datasets.load_iris()
_X = iris.data
y = iris.target

#make 2-D array of target variables
df_y = [[int(target)] for target in y]

#concatenate X array and df_y array in one row with all 5 columns
data = np.concatenate((_X, df_y), axis=1)
#print(data)

#define types
types_dict = {0:'Setosa', 1:'Versicolour', 2:'Virginica'}
#define columns
columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Type']

#create a DataFrame with those data
dataset = pd.DataFrame(data=data, columns=columns)
#dataset.to_csv('./data-iris.csv', index=False)

pca = PCA(n_components=2)
X = pca.fit_transform(_X)
##########################################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=36851234)

tpot = TPOTClassifier(generations=10, population_size=50, verbosity=2, random_state=42, cv=rskf, scoring='f1_micro')

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_iris_pipeline.py')
