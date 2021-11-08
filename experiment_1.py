import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from model.onevsall import OneVsAllClassifier
from model.adaboost import BinaryAdaBoost
from model_selection.nested_cv import nestedCV
from model.adaboost import MultiClassAdaBoost


if __name__ == "__main__":
    
    # read data
    data = pd.read_csv("data/covtype.csv")
    X = data.drop("Cover_Type", axis=1).values
    y = data["Cover_Type"].values - 1
    
    # extract sample
    Xsample, _, Ysample, _ = train_test_split(X, y, stratify=y, train_size=10000, random_state=0)
    
    # statistic over label
    label, label_counts = np.unique(Ysample, return_counts=True)
    for l, c in zip(label, label_counts):
        print(f"Label {l} has {c/label_counts.sum()}% ({c}) examples")
        
    # costant classifier: majority vote
    majority_label = label_counts.argmax()
    costant_classifier_score = (Ysample!=majority_label).mean()
    print(f"\nMajority label classifier zero-one loss: {costant_classifier_score}")
    
    # One Vs all AdaBoost decision stump - nested cross validation to tune T
    model = OneVsAllClassifier(BinaryAdaBoost())
    params = [{"T":1}, {"T":101}, {"T": 301}]
    print("\nNested CV - One Vs All AdaBoost decision stump")
    print(f"hyper-parameters: {params}")
    train_acc, test_acc = nestedCV(model, params,  Xsample, Ysample, n_external_fold=3, n_internal_fold=3, verbose=True)
    print(f"train loss {1-train_acc} | test loss {1-test_acc}")
    
    model = MultiClassAdaBoost(K=7)
    print("\nNested CV - MH.AdaBoost decision stump")
    print(f"hyper-parameters: {params}")
    train_acc, test_acc = nestedCV(model, params,  Xsample, Ysample, n_external_fold=3, n_internal_fold=3, verbose=True)
    print(f"train loss {1-train_acc} | test loss {1-test_acc}")
    
    print("\nNested CV - SKLEARN One Vs All AdaBoost decision stump ")
    print(f"hyper-parameters: {params}")
    model = GridSearchCV(OneVsRestClassifier(AdaBoostClassifier()), 
                         param_grid = {"estimator__n_estimators": [1, 101, 301]}, cv=3, verbose=1)
    d_accuracy = cross_validate(model, Xsample, Ysample, cv=3, verbose=1, return_train_score=True)
    print(f"train loss {1-d_accuracy['train_score'].mean()} | test loss {1-d_accuracy['test_score'].mean()}")
    
    
    
    print("\nNested CV - SKLEARN Multiclass AdaBoost decision stump ")
    print(f"hyper-parameters: {params}")
    model = GridSearchCV(AdaBoostClassifier(), 
                         param_grid = {"n_estimators": [1, 101, 301]}, cv=3, verbose=1)
    d_accuracy = cross_validate(model, Xsample, Ysample, cv=3, verbose=1, return_train_score=True)
    print(f"train loss {1-d_accuracy['train_score'].mean()} | test loss {1-d_accuracy['test_score'].mean()}")
