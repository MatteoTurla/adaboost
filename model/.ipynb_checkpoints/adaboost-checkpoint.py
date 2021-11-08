import numpy as np
from model.decisionstump import BinaryDecisionStump, MultiClassDS

class BinaryAdaBoost:
    """
    Binary adaboost classifier
    
    Attributes
    ----------
    T: int
        number of boosting rounds
    wls: list[BinaryDecisionStump]
    w: list[float]
        weights used to compute the final ensemble of predictors
        
    Methods
    -------
    fit(X,y,D, utils_dict)
    predict(X)
        return sign of the weighted binary prediction
    decision_function(X) 
        return the weighted sum of binary prediciton
    clone()
    score(X, y)
        compute 1 - zero-one loss function, equivalent of accuracy score
    set_params(**dict)
        set new params used by croass-validation methods
        
    """
    def __init__(self, base_estimator=BinaryDecisionStump(), T=50):
        """
        Parameters
        ----------
        T: int 
            number of boosting rounds
        """
        self.T = T
        self.base_estimator = base_estimator
        
        # default init
        self.wls = []
        self.w = []
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            p = parameter.split(".")
            if len(p)==1:
                setattr(self, parameter, value)
            else:
                setattr(self.base_estimator, p[1], value)
        return self
        
    
        
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
        D = np.ones(nrow)/nrow
        
        utils_dict = self.base_estimator.preprocessing(X)
        
        for t in range(self.T):
            
            wl = self.base_estimator.clone()
            wl.fit(X, y, D, utils_dict)
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
        compute sign of weighted sum of binary prediction of each weak predictor
        
        Parameters
        ----------
        X: m by d matrix
        """
        return np.sign(sum([w*wl.predict(X) for w, wl in zip(self.w, self.wls)]))
    
    def decision_function(self, X):
        """
        compute weighted sum of binary prediction of each weak predictor
        
        Parameters
        ----------
        X: m by d matrix
        """
        
        return sum([w*wl.predict(X) for w, wl in zip(self.w, self.wls)])
            
    def score(self, X, y):
        """
        compute 1 - zero-one loss function, equivalent of accuracy score
        
        Parameters
        ----------
        X: n by d matrix
        y: column vector of size n
            true label associated to each example
        
        Output
        ------
        accuracy: float in (0,1)
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        
        return accuracy
    
    def clone(self):
        """
        return an unfitted clone of the current object 
        """
        return BinaryAdaBoost(self.base_estimator.clone(), self.T)
        

        
        
class MultiClassAdaBoost:
    """
    Binary adaboost classifier
    
    Attributes
    ----------
    T: int
        number of boosting rounds
    wls: list[BinaryDecisionStump]
    w: list[float]
        weights used to compute the final ensemble of predictors
        
    Methods
    -------
    fit(X,y,D, utils_dict)
    predict(X)
        return sign of the weighted binary prediction
    decision_function(X) 
        return the weighted sum of binary prediciton
    clone()
    score(X, y)
        compute 1 - zero-one loss function, equivalent of accuracy score
    set_params(**dict)
        set new params used by croass-validation methods
        
    """
    def __init__(self, K, T=50):
        """
        Parameters
        ----------
        T: int 
            number of boosting rounds
        """
        self.K = K
        self.T = T
        
        # default init
        self.wls = []
        self.w = []
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
        
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
            e /= self.K
            
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
            
    def predict(self, X):
        """
        compute sign of weighted sum of binary prediction of each weak predictor
        
        Parameters
        ----------
        X: m by d matrix
        """
        predictions = [sum([alpha*model.predict(X, l) for model, alpha in zip(self.wls, self.w)]) for l in range(self.K)]
        return np.array(predictions).T.argmax(axis=1)
           
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        
        return accuracy
    
    def clone(self):
        return MultiClassAdaBoost(K=self.K, T=self.T)