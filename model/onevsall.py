import numpy as np

class OneVsAllClassifier:

    def __init__(self, estimator):
        self.estimator = estimator
        self.models = [] 
        
    def set_params(self, **parameters):
        self.estimator.set_params(**parameters)
        return self
        
    def fit(self, X, y):

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
            
    def predict(self, X):
        score = np.array([model.decision_function(X) for model in self.models]).T
        return score.argmax(axis=1)
    
    def score(self, X, y):
        return np.array(self.predict(X) == y).mean()
    
    def clone(self):
        return OneVsAllClassifier(self.estimator.clone())