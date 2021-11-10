from sklearn.model_selection import StratifiedKFold
import numpy as np

def GridSearchCV(classifier, params, X, y, n_fold, verbose=False):
    """
    Exhaustive search over specified parameter values for a classifier. 
    For each set of parameter in the grid, cross validation is used to estimate the performance of the classifier
    
    Input
    -----
    classifier
    params: list of dict
        each dict contains the set of hyperparameters to use to train the model
    X: training datapoints
    y: training labels
    n_fold: integer
        number of folds to use in the cross-validation
    verbose: boolean
        if True print cross-validation stats
        
    Output
    ------
    best_params:
        dict containg the best hyperparameters in the grid
    best_scores:
        cv scores achived by best_params
    """
    if verbose:
        print(f"\nGrid Search CV with {n_fold} folds")
        print(f"Grid: {params}")
    skf = StratifiedKFold(n_splits=n_fold)
    scores = []
    for param in params:
        train_cv_score = 0
        val_cv_score  = 0
        for train_index, val_index in skf.split(X, y):
            classifier = classifier.clone()
            classifier.set_params(**param)
        
            classifier.fit(X[train_index, :], y[train_index])
            
            train_cv_score += classifier.score(X[train_index, :], y[train_index])
            val_cv_score += classifier.score(X[val_index, :], y[val_index])
            
        train_cv_score /= n_fold
        val_cv_score /=  n_fold
        
        if verbose:
            print("-"*10)
            [print(key, " : ", value) for key, value in param.items()]
            print(f"train score: {train_cv_score}")
            print(f"validation score {val_cv_score}")
            
        scores.append(val_cv_score)
        
    argmax = np.array(scores).argmax()
    if verbose:
        print(f"Grid Search CV -> best validation score: {scores[argmax]} with parameters {params[argmax]}")
    return (params[argmax], scores[argmax])

