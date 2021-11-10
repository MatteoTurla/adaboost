import numpy as np
from model.base_classifier import BaseClassifier

class OneVsAllClassifier(BaseClassifier):
    """
    One Vs All classifier
    
    Attributes
    ----------
    estimator: Classifier Object
    """

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.models = [] 
        
    def set_params(self, **parameters):
        self.estimator.set_params(**parameters)
        return self
        
    def preprocessing(self, X):
        return {}
            
    def fit(self, X, y, D=None, kwargs=None):

        nrow, ncol = X.shape
        
        labels = np.unique(y)
        n_labels = len(labels)
        
        for k in range(n_labels):
            
            # create one vs all vector of label
            yk = np.ones(nrow)
            yk[y!=k] = -1
            
            model = self.estimator.clone()
            model.fit(X, yk)
            self.models.append(model)
            
        return self
    
    def decision_function(self, X):
        score = np.array([model.decision_function(X) for model in self.models]).T
        return score
            
    def predict(self, X):
        score = np.array([model.decision_function(X) for model in self.models]).T
        return score.argmax(axis=1)
    
    def clone(self):
        return OneVsAllClassifier(self.estimator.clone())