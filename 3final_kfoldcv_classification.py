# Try different Classifiers

parent_dir = ''

this_dir = './'
plot_dir = 'algorithms-comparison/'
path_to_plot_dir = parent_dir+this_dir+plot_dir
from pathlib import Path
Path(path_to_plot_dir).mkdir(parents=True, exist_ok=True)

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from evaluate import *
import operator

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

pca = PCA(n_components=2)
X = pca.fit_transform(_X)

#########################################################################################################

logit = LogisticRegression(class_weight='balanced')
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
ada = AdaBoostClassifier()
rf = RandomForestClassifier(class_weight='balanced')
nb = GaussianNB()
svm = SVC(probability=True, class_weight='balanced')
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

models = {'logistic regression':logit, 'k-nearest neighbors':knn, 'adaboost':ada,
           'random forest':rf, 'naive bayes':nb, 'support vector machines':svm,
           'linear discriminant analysis': lda, 'quadratic discriminant analysis':qda}

metrics_functions = {
'accuracy':get_balanced_accuracy,
'f1':get_micro_f1,
'hamming loss':get_hamming_loss
}

metrics_values = {
'accuracy': {},
'f1': {},
'hamming loss': {}
}

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

print('Using Repeated Stratified K-Fold Cross Validation.\n')
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=36851234)

names = []
results = {
'accuracy': {},
'f1': {},
'hamming loss': {}
}

for name, model in sorted(models.items()):
    scores = []
    mean_score = 0.0
    std_scores = 0.0
    names.append(name)
    for metrics_name, metrics_func in sorted(metrics_functions.items()):
        scores = cv_evaluate_model(rskf, X, y, model, metrics_func)
        mean_score = np.mean(scores)
        std_scores = np.std(scores)
        metrics_values[metrics_name].update({name: (mean_score,std_scores)})
        results[metrics_name].update({name: scores})
    print('\n')

for metric, values in sorted(results.items()):
    plt.title(metric.capitalize())
    plt.boxplot([values[model] for model in names], labels=names, showmeans=True)
    plt.xticks(rotation=90)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path_to_plot_dir+metric+'.png', dpi=250)
    plt.clf()
    plt.close()

for metrics, values_per_model in sorted(metrics_values.items()):
    reverse = (metrics != 'hamming loss')
    sorted_values_per_model = sorted(values_per_model.items(), key=operator.itemgetter(1), reverse=reverse)
    for i, item in enumerate(sorted_values_per_model):
        print(str(i+1)+') Model number ' + str(i+1) + ' according to ' + metrics.upper())
        print(item[0]+':{:0.4f} ± {:0.4f}'.format(item[1][0],item[1][1]))
    print('\n')
