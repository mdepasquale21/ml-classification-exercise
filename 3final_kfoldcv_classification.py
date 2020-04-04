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
from sklearn.model_selection import RepeatedStratifiedKFold
from evaluate import *
import operator

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

################################################################################################################################
print('DATA EXPLORATION')
print('SHAPE')
print(dataset.shape)
print('INFO')
dataset.info()
print('DESCRIPTION')
print(dataset.describe())
n_rows_head = 5
print('FIRST ' + str(n_rows_head) + ' ENTRIES')
print(dataset.head(n_rows_head))
print('\n')
################################################################################################################################

logit = LogisticRegression(class_weight='balanced')
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
ada = AdaBoostClassifier()
rf = RandomForestClassifier(class_weight='balanced')
nb = GaussianNB()
svm = SVC(probability=True, class_weight='balanced')

models = {'logistic regression':logit, 'k-nearest neighbors':knn, 'adaboost':ada,
           'random forest':rf, 'naive bayes':nb, 'support vector machines':svm}

metrics_functions = {
'accuracy':get_accuracy,
'f1':get_f1,
'f05':get_f05,
'precision':get_precision,
'recall':get_recall,
'auc roc':get_roc_auc,
'auc pr':get_pr_auc,
'hamming loss':get_hamming_loss
}

metrics_values = {
'accuracy': {},
'f1': {},
'f05': {},
'precision': {},
'recall': {},
'auc roc': {},
'auc pr': {},
'hamming loss': {}
}

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

print('Using Repeated Stratified K-Fold Cross Validation.\n')
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=36851234)

names = []
results = {
'accuracy': {},
'f1': {},
'f05': {},
'precision': {},
'recall': {},
'auc roc': {},
'auc pr': {},
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
