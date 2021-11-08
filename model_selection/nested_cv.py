from sklearn.model_selection import StratifiedKFold
import numpy as np
from model_selection.cv import GridSearchCV

def nestedCV(classifier, params, X, y, n_external_fold, n_internal_fold, verbose=False):
    """
    nested CV to estimate the performance of a learning algorithm which requires to tune its hyper-parameters.
    to tune the hyper-parameters it is used an internal grid search cross-validation.
    to estimate the performance of the learning algoritm it used external cross validation.
    
    Input
    -----
    classifier
    params: list of dict
        each dict contains the set of hyperparameters to use to train the model
    X: training datapoints
    y: training labels
    n_external_fold: integer
        number of folds to use in external cross validation
    n_internal_fold: integer
        number of folds to use in intertnal grid search cross validation
    verbose: boolean
        if True print stats 
        
    Output:
    train_score: external cross validation accuracy on training sets
    test_score: external cross validation accuracy on validation sets
    """
    skf = StratifiedKFold(n_splits=n_external_fold)
    
    train_score = 0
    test_score = 0
    
    for fold_number, (train_index, test_index) in enumerate(skf.split(X, y)):
        best_param, internal_score = CV(classifier, params, X[train_index,:], y[train_index], n_internal_fold, verbose=verbose)
        
        classifier = classifier.clone()
        classifier.set_params(**best_param)
        
        classifier.fit(X[train_index, :], y[train_index])
        
        trs = classifier.score(X[train_index,:], y[train_index])
        tts = classifier.score(X[test_index,:], y[test_index])
        
        train_score += trs
        test_score += tts
        
        if verbose:
            print(f"Fold number: {fold_number}, score: {tts}, best params: {best_param}")
    
    train_score /= n_external_fold
    test_score /= n_external_fold
    return train_score, test_score