import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, hamming_loss
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import make_scorer

def get_balanced_accuracy(y_true, y_pred):
    """
    This function just wraps sklearn balanced accuracy score.
    """
    return balanced_accuracy_score(y_true, y_pred)

def get_micro_f1(y_true, y_pred):
    """
    This function just wraps sklearn f1 score.
    """
    return f1_score(y_true, y_pred, average='micro')

def get_macro_f1(y_true, y_pred):
    """
    This function just wraps sklearn f1 score.
    """
    return f1_score(y_true, y_pred, average='macro')

def get_micro_precision(y_true, y_pred):
    """
    This function just wraps sklearn precision score.
    """
    return precision_score(y_true, y_pred, average='micro')

def get_macro_precision(y_true, y_pred):
    """
    This function just wraps sklearn precision score.
    """
    return precision_score(y_true, y_pred, average='macro')

def get_micro_recall(y_true, y_pred):
    """
    This function just wraps sklearn recall score.
    """
    return recall_score(y_true, y_pred, average='micro')


def get_macro_recall(y_true, y_pred):
    """
    This function just wraps sklearn recall score.
    """
    return recall_score(y_true, y_pred, average='macro')

def get_hamming_loss(y_true, y_pred):
    """
    This function just wraps sklearn hamming loss.
    """
    return hamming_loss(y_true, y_pred)

def print_all_metrics(y_true,y_pred):
    """
    This function prints all metrics.
    """
    print('BALANCED ACCURACY')
    accuracy = get_balanced_accuracy(y_true, y_pred)
    print(accuracy)
    print('MICRO F1')
    print(get_micro_f1(y_true, y_pred))
    print('MACRO F1')
    print(get_macro_f1(y_true, y_pred))
    print('MICRO PRECISION')
    print(get_micro_precision(y_true, y_pred))
    print('MACRO PRECISION')
    print(get_macro_precision(y_true, y_pred))
    print('MICRO RECALL')
    print(get_micro_recall(y_true, y_pred))
    print('MACRO RECALL')
    print(get_macro_recall(y_true, y_pred))
    print('HAMMING LOSS')
    hamming_loss = get_hamming_loss(y_true, y_pred)
    print(hamming_loss)

def print_some_metrics(y_true,y_pred):
    """
    This function prints some metrics.
    """
    print('BALANCED ACCURACY')
    accuracy = get_balanced_accuracy(y_true, y_pred)
    print(accuracy)

    print('MICRO F1')
    print(get_micro_f1(y_true, y_pred))

    print('MICRO PRECISION')
    print(get_micro_precision(y_true, y_pred))

    print('MICRO RECALL')
    print(get_micro_recall(y_true, y_pred))

    print('HAMMING LOSS')
    hamming_loss = get_hamming_loss(y_true, y_pred)
    print(hamming_loss)

def calc_learning_curve(cv, X, y, model, metrics_func):
    """
    This function evaluates a model's learning curve using a given Cross Validation with a
    given metrics function.
    """
    # define the model evaluation metric
    metric = make_scorer(metrics_func)
    # evaluate learning curve
    # learning_curve returns train_sizes, train_scores, test_scores
    # and fit_times, score_times only if return_times=True (default return_times=False)
    # using default value for train_sizes that is np.linspace(0.1, 1.0, 5)
    lc = learning_curve(estimator=model, X=X, y=y, cv=cv, scoring=metric)
    return lc

def cv_evaluate_model(cv, X, y, model, metrics_func):
    """
    This function evaluates a model using a given Cross Validation with a
    given metrics function.
    """
    # define the model evaluation metric
    metric = make_scorer(metrics_func)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
    return scores
