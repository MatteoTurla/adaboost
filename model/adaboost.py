import numpy as np
from model.decisionstump import BinaryDecisionStump, MultiClassDS
from model.base_classifier import BaseClassifier

class BinaryAdaBoost(BaseClassifier):
    """
    Binary adaboost classifier
    class is implemented to be able to work also with other base classifier 
    
    Attributes
    ----------
    T: int
        number of boosting rounds
    wls: list[BinaryDecisionStump]
    w: list[float]
        weights used to compute the final ensemble of predictors
    """
    def __init__(self, base_estimator=BinaryDecisionStump(), T=50):
        """
        Parameters
        ----------
        base_estimator: BaseClassifier derived class
            base classifier to use as learning algorithm
        T: int 
            number of boosting rounds
        """
        super().__init__()
        self.T = T
        self.base_estimator = base_estimator
        
        # default init
        self.wls = []
        self.w = []
    
    def preprocessing(self, X):
        return {}
             
    def fit(self, X, y, D=None, kwargs=None):
        """
        Learn AdaBoost classifier
        """
        
        nrow, ncol = X.shape
        D = np.ones(nrow)/nrow
        
        # if base estimatore requires some preprocessing before running the fit model it is done one time instead of T times
        kwargs = self.base_estimator.preprocessing(X)
        
        for t in range(self.T):
            
            wl = self.base_estimator.clone()
            wl.fit(X, y, D, kwargs)
            y_pred = wl.predict(X)
            L = y_pred*y
            
            e = np.sum(D[L <= 0])
            
            # special case: ls(h) = 1/2
            if abs(e-0.5) <= 1e-10:
                return self
            
            # special case: ls(h) = 0 or ls(h) = 1
            if abs(e) <= 1e-5 or abs(e-1) <= 1e-5:
                self.wls = [wl]
                self.w = [1]
                return self
              
            w = 0.5*np.log((1-e)/e)
            self.wls.append(wl)
            self.w.append(w)
            
            D = ( D*np.exp(-1*L*w) ) 
            D = D/np.sum(D)
        return self
            
    def predict(self, X):
        """
        f(x) = sign(sum w_i * h_i(x)) where i = 1, ..., T
        """
        return np.sign(sum([w*wl.predict(X) for w, wl in zip(self.w, self.wls)]))
    
    def decision_function(self, X):
        """
        F(x) = sum w_i * h_i(X) where i = 1, ..., T
        """
        return sum([w*wl.predict(X) for w, wl in zip(self.w, self.wls)])
            
    def clone(self):
        return BinaryAdaBoost(self.base_estimator.clone(), self.T)
        

        
        
class MultiClassAdaBoost(BaseClassifier):
    """
    MultiLabel AdaBoost classifier (AdaBoost.MH)
    
    Attributes
    ----------
    T: int
        number of boosting rounds
    wls: list[BinaryDecisionStump]
    w: list[float]
        weights used to compute the final ensemble of predictors 
    """
    def __init__(self, K, T=50):
        """
        Parameters
        ----------
        T: int 
            number of boosting rounds
        """
        super().__init__()
        self.K = K
        self.T = T
        
        # default init
        self.wls = []
        self.w = []
        
    def preprocessing(self, X):
        return {}
    
    def fit(self, X, y):
        """
        Learn Binary Decision stumps and associated weights
        
        Parameters
        ----------
        X: m by d matrix
            matrix of examples
        y: vector of size m which elements are either 1 or -1
        """
        
        nrow, ncol = X.shape
        D = np.ones((nrow, self.K))/(nrow*self.K)
        Y = np.ones((nrow, self.K))*(-1)
        Y[(range(nrow), y)] = 1
        
        utils_dict = MultiClassDS(self.K).preprocessing(X)
        
        for t in range(self.T):
            
            wl = MultiClassDS(self.K)
            wl.fit(X, Y, D, utils_dict)
            
            e = 0
            for k in range(self.K):
                y_k = wl.predict(X, k)
                e += D[y_k != Y[:,k], k].sum()
            
            # special case: ls(h) = 1/2
            if abs(e-0.5) <= 1e-10:
                return self
            
            # special case: ls(h) = 0 or ls(h) = 1
            if abs(e) <= 1e-5 or abs(e-1) <= 1e-5:
                self.wls = [wl]
                self.w = [1]
                return self
              
            w = 0.5*np.log((1-e)/e)
            self.wls.append(wl)
            self.w.append(w)
            
            for k in range(self.K):
                D[:, k] = D[:,k]*np.exp(-1*w*Y[:,k]*wl.predict(X, k))
            D = D/np.sum(D)
                        
        return self
    
    def decision_function(self, X):
        confidence = [sum([alpha*model.predict(X, l) for model, alpha in zip(self.wls, self.w)]) for l in range(self.K)]
        return np.array(predictions).T
            
    def predict(self, X):
        """
        f(x) = argmax(l = 1,.., K) sum w_i(l)*h_i(x, l) where i = 1, ..., T
        """
        predictions = [sum([alpha*model.predict(X, l) for model, alpha in zip(self.wls, self.w)]) for l in range(self.K)]
        return np.array(predictions).T.argmax(axis=1)
 
    def clone(self):
        return MultiClassAdaBoost(K=self.K, T=self.T)