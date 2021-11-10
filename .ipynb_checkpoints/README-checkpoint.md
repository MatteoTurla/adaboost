# AdaBoost and Decision Stump from scratch

Project strcuture:

```
project
│   README.md
│   experiment_1.py         evaluates performances of implemented learning algorithms using nested cv
│   _bias_variance.ipynb    bias-variance tradeoff in selecting different base estimator
│   _slow_vs_fast.ypnb      compute time of iterative and cumsum implmementation of Decion Stump
│   _training_time.ypinb    compute training time of Binary AdaBoost and compare it with SKLEARN implementation
│
└───data     contains data used in experiments
└───figure   contains figures used in the paper
└───model
│   │   base_classifier.py    abstract class from which all the implemented learning algorithms inherit
│   │   adaboost.py           concreate class implementation of AdaBoost learning algorithm for binary and multiclass classification
│   │   decisionstump.py      concrete class implementation of DecisionStump for binary and multiclass classification
│   │   onevsall.py           concrete class implmentation of one vs all classifier. it accepts any BaseClassifier implementation
│   │
└───model    contains functions used to estimate the performance of any learning algorithm which implement BaseClassifier
│   │   cv.py                 grid search CV to tune hyper parameters
│   │   nested_cv.py          nested cv to estimate performance of any learning algorithm
│   │
└───old
    │   decisionstump.py      a slow implementation of Decion Stump based on iteration
    │   
└───pdf_notebook_output    contains pdf output of all used notebooks and results of experiment_1.py
```


 
