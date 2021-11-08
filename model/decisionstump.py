import numpy as np
    
class BinaryDecisionStump:
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
        
    Methods
    -------
    fit(X, Y, D, utils_dict)
        learn theta, j
    predict(X)
        return vector of prediction in {-1, 1}, one for each example
    score(X, y)
        compute 1 - zero-one loss function, equivalent of accuracy score
    """
    def __init__(self):
        self.j = None
        self.theta = None
        
    def preprocessing(self, X):
        """
        compute the utils dictionary passed to the decision stump
        
        Parameters
        ----------
        X: m by d matrix
        
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
                
    def fit(self, X, y, D, utils_dict):
        """
        Learn theta, j, b. faster version of pseudocode explained in 'Undertand machine learning: from theoy to algorithms'
        the predictor is the ERM with respect to the zero-one-loss weigthed by D
        
        Parameters
        ----------
        X: m by d matrix
            matrix of examples
        y: vector of size m which elements are either 1 or -1
        D: vector of size m which element are in (0,1) and sum up to 1
        utils_dict: {'sorted_index': list[int][int], 'unique_index': list[int][int]}
            passed by Adaboost to speed up computation over T boosting rounds
            sorted_index[col]: contains for each column the index of the sorted value of X[:, col]
            unique_index[col]: contains for each column the index of the unique value in X[:, col]
        """
        
        nrow, ncol = X.shape
        F_star = 0 
        
        for col in range(ncol):
            
            sorted_index = utils_dict[col]["sorted_index"]
            # ordering X[:,col], y, D
            sorted_val = X[sorted_index, col]     
            sorted_y = y[sorted_index]
            sorted_D = D[sorted_index]
            
            # indeces used to optimize objective function
            obj_index = utils_dict[col]["unique_index"]
            
            # init h -> all examples classified as -1, F is the weighted zero-one loss
            F = np.sum(sorted_D[sorted_y==1]) 
                   
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
                 
    def predict(self, X):
        """
        compute prediction for each example in X
        
        h(x) = sign(theta - x[j])*b
        
        Parameters
        ----------
        X: n by d matrix of examples
        
        Output
        ------
        column vector of size n
        """
        return np.sign(self.theta - X[:, self.j] + 1e-8)
    
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
        return BinaryDecisionStump()
    
class MultiClassDS:
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
    
    Methods
    -------
    fit(X,Y)
        fit K binary decision stump
    predict(X, l)
        binary prediction of examples X using l_th decision stump. if h(x)==1 then example x predicted with label l.
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
        compute the utils dictionary passed to the decision stump
        
        Parameters
        ----------
        X: m by d matrix
        
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
        
    def fit(self, X, Y, D, utils_dict):
        """
        it learns K binary decision stumps
        
        Parameters
        ----------
        X: m by d matrix
            matrix of examples
        Y: m by K matrix, each element in {-1, 1}
            Y(i,j) = 1 iff label(x_i) = j
                    -1 otherwise
         D: m by K matrix
        utils_dict: {'sorted_index': list[int][int], 'unique_index': list[int][int]}
            passed by Adaboost to speed up computation over T boosting rounds
            sorted_index[col]: contains for each column the index of the sorted value of X[:, col]
            unique_index[col]: contains for each column the index of the unique value in X[:, col]
        """
        for k in range(self.K):
            dt = BinaryDecisionStump()
            dt.fit(X, Y[:, k], D[:, k], utils_dict)
            self.DecisionStumps.append(dt)
    
    def predict(self, X, l):
        """
        binary prediction of examples X using l_th decision stump. if h(x)==1 then example x predicted with label l.
        
        Parameters:
        X: m by d matrix
        l: int
            binary classification over label l
            output 1 means example x is associated with label l, -1 otherwise
        """
        return self.DecisionStumps[l].predict(X)
    