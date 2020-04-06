import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file -> no, manually changed to 'Type' in this script
tpot_data = pd.read_csv('../data-iris.csv') #manually removed separator and dtype, handling them by myself!
output = tpot_data['Type'].astype(int)
features = tpot_data.drop('Type', axis=1)

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, output, random_state=42)

# Average CV score on the training set was: 0.9527027027027025
exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.45),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.15000000000000002, min_samples_leaf=8, min_samples_split=16, n_estimators=100)),
    KNeighborsClassifier(n_neighbors=31, p=1, weights="distance")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('RESULTS OF THE TPOT PIPELINE:')
print(results)
