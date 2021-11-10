import numpy as np
from model.base_classifier import BaseClassifier

class BinaryDecisionStump(BaseClassifier):
    """
    Binary DecisionStump: H: X -> {-1, 1}
    
    h(x) = sign(theta - x[j])*b
    
    note: we do not need to optimize for b. Adaboost will reverse the prediction if needed.
    The goal is to find the split whose error maximize the distance from a random classifier.
    
    Attributes
    ----------
    theta: float
        decision value
        learnt by fit function
    j: int
        decision column, index of feature used to predict label
        learnt by fit function
    """
    def __init__(self):
        super().__init__()
        self.j = None
        self.theta = None
        
    def preprocessing(self, X):
        """
        For each attributes sort the array and compute unique value.
        
        Output
        ------
        utils_dict: {'sorted_index': list[int][int], 'unique_index': list[int][int]}
            sorted_index[col]: contains for each column the index of the sorted value of X[:, col]
            unique_index[col]: contains for each column the index of the unique value in X[:, col]        
        """
        nrow, ncol = X.shape
        
        utils_dict = {}
        for col in range(ncol):
            sorted_index = np.argsort(X[:, col])
            u,i,c = np.unique(X[sorted_index, col], return_index=True, return_counts=True)
            unique_index = i + (c-1)
            utils_dict[col] = {"sorted_index": sorted_index, "unique_index": unique_index}
        return utils_dict
                
    def fit(self, X, y, D, kwargs):
        """
        Learn theta, j, b. faster version of pseudocode explained in 'Undertand machine learning: from theoy to algorithms'
        the predictor is the ERM with respect to the zero-one-loss weigthed by D
        """
        
        nrow, ncol = X.shape
        F_star = 0 
        
        for col in range(ncol):
            
            sorted_index = kwargs[col]["sorted_index"]
            # ordering X[:,col], y, D
            sorted_val = X[sorted_index, col]     
            sorted_y = y[sorted_index]
            sorted_D = D[sorted_index]
            
            # indeces used to optimize objective function
            obj_index = kwargs[col]["unique_index"]
            
            # init h -> all examples classified as -1, F is the weighted zero-one loss
            F = np.sum(sorted_D[sorted_y==1]) 
            f_obj = np.abs(F-0.5)
            if f_obj > F_star:
                self.j = col
                self.theta = sorted_val[0] - 0.5
                F_star = f_obj
            
                   
            # compute objective  value over all possible split of current column 
            cs = np.cumsum(-1*sorted_D*sorted_y)
            f_obj = np.abs((F+cs) - 0.5) 
            # get best split of current column
            argmax = np.argmax(f_obj[obj_index])
            nonconsecutive_argmax = obj_index[argmax]
            
            # compare best split of current column with best split achived until now
            if f_obj[nonconsecutive_argmax] > F_star:
                # update theta, j, b
                F_star = f_obj[nonconsecutive_argmax]
                self.j = col
                self.theta = sorted_val[nonconsecutive_argmax]
                
    def decision_function(self, X):
        return self.theta - X[:, self.j] + 1e-8
                 
    def predict(self, X):
        """
        h(x) = sign(theta - x[j])*b
        """
        return np.sign(self.theta - X[:, self.j] + 1e-8)
        
    def clone(self):
        return BinaryDecisionStump()
    
class MultiClassDS(BaseClassifier):
    """
    MultiClass decision stump, it learns K decision stump
        
    for k in {0, ..., K-1} it creates a decision stump minimizing the zero-one loss over Y[:,k] weighted by D[:, k] where k is the
    the positive label in the oneVsAll setting.
    
    The resulting predictor is a list of K decisionStump , one for each unique label.
    
    Attributes
    ----------
    K: int 
        number of unique labels
    DecisionStumps: list[BinaryDecisionStump]
        list of K binary decision stump
        learnt by fit function
    """
    
    def __init__(self, K):
        """
        Parameters
        ----------
        K: int
            number of unique labels
        """
        self.K = K # number of unique labels
        
        # learnt by fit function
        self.DecisionStumps = []
        
    def preprocessing(self, X):
        """
        For each attributes sort the array and compute unique value.
        
        Output
        ------
        utils_dict: {'sorted_index': list[int][int], 'unique_index': list[int][int]}
            sorted_index[col]: contains for each column the index of the sorted value of X[:, col]
            unique_index[col]: contains for each column the index of the unique value in X[:, col]        
        """
        nrow, ncol = X.shape
        
        utils_dict = {}
        for col in range(ncol):
            sorted_index = np.argsort(X[:, col])
            u,i,c = np.unique(X[sorted_index, col], return_index=True, return_counts=True)
            unique_index = i + (c-1)
            utils_dict[col] = {"sorted_index": sorted_index, "unique_index": unique_index}
        return utils_dict
        
    def fit(self, X, Y, D, kwargs):
        """
        it learns K binary decision stumps
        """
        for k in range(self.K):
            dt = BinaryDecisionStump()
            dt.fit(X, Y[:, k], D[:, k], kwargs)
            self.DecisionStumps.append(dt)
            
    def decision_function(self, X):
        return self.DecisionStumps[l].decision_function(X)
    
    def predict(self, X, l):
        """
        binary prediction of examples X using l_th decision stump. if h(x)==1 then example x predicted with label l.
        """
        return self.DecisionStumps[l].predict(X)
    
    def clone(self):
        return MultiClassDS(self.K)
    