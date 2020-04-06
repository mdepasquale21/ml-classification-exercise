import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA

#import data
iris = datasets.load_iris()
X = iris.data
y = iris.target

#make 2-D array of target variables
df_y = [[int(target)] for target in y]

#concatenate X array and df_y array in one row with all 5 columns
data = np.concatenate((X, df_y), axis=1)
#print(data)

#define types
types_dict = {0:'Setosa', 1:'Versicolour', 2:'Virginica'}
#define columns
columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Type']

#create a DataFrame with those data
dataset = pd.DataFrame(data=data, columns=columns)

dataset.to_csv('./data-iris.csv', index=False)

################################################################################################################################

#explore dataframe
print('\nDATA EXPLORATION')
print('\nSHAPE')
print(dataset.shape)
print('\nINFO')
dataset.info()
print('\nDESCRIPTION')
print(dataset.describe())
n_rows_head = 10
print('\nFIRST ' + str(n_rows_head) + ' ENTRIES')
print(dataset.head(n_rows_head))
print('\nMINIMUM VALUES')
print(dataset.min())
print('\nMAXIMUM VALUES')
print(dataset.max())
print('\nMEAN VALUES')
print(dataset.mean())

################################################################################################################################

#Heatmap
plt.subplots(figsize=(13,10))
heat = dataset.corr()
sns.heatmap(heat)
sns.heatmap(heat, annot = True)
plt.yticks(rotation=0)
plt.savefig('iris_dataset_heatmap.png', dpi = 250)
plt.clf()
plt.close()

################################################################################################################################
################################################################################################################################

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("\nEXPLAINED VARIANCE BY FIRST 2 PRINCIPAL COMPONENTS:")
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

# create scatter plot
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Plot of first 2 PCA directions')
for i, j in enumerate(np.unique(y)):
        plt.scatter(X_reduced[y == j, 0], X_reduced[y == j, 1],
                    c = [matplotlib.colors.ListedColormap(('red', 'green', 'orange'))(i)], label = j)
plt.legend((types_dict[0], types_dict[1], types_dict[2]),loc='upper right', bbox_to_anchor=(1.05, 1.17), ncol=3)
plt.savefig('iris-data-scatter.png', dpi = 250)
plt.clf()
plt.close()

################################################################################################################################
################################################################################################################################

print('\nNUMBER OF ENTRIES PER LABEL')
print(dataset.groupby('Type').size())
print('\n')

kde_style = {"color": "darkcyan", "lw": 2, "label": "KDE", "alpha": 0.7}
hist_style = {"histtype": "stepfilled", "linewidth": 3, "color":"darkturquoise", "alpha": 0.25}

# histogram of y values
sns.distplot(dataset.Type, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Type of Iris Histogram')
plt.xlabel('Type')
plt.ylabel('Frequency')
#plt.axvline(dataset.Type.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2) #nonsense in this case
plt.savefig('iris_type_histogram.png', dpi = 250)
plt.clf()
plt.close()

################################################################################################################################
################################################################################################################################

plt.rcParams.update({'font.size': 8})

#pairwise scatterplot matrix
pd.plotting.scatter_matrix(dataset)
plt.savefig('iris-scatterplot-matrix.png', dpi = 250)
plt.clf()
plt.close()
