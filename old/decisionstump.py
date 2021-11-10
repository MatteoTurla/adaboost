import numpy as np

class DecisionStump:
    """
    slow version of decision stump based on iteration.
    """
    def __init__(self):
        self.j = None
        self.theta = None
        self.b = None
                
    def fit(self, X, y, D):
        

        nrow, ncol = X.shape
        F_star = 1
        
        for col in range(ncol):
            
            sorted_index = np.argsort(X[:, col])
            sorted_val = X[sorted_index, col]     
            sorted_val = np.insert(sorted_val, nrow, sorted_val[-1] + 1)
            sorted_y = y[sorted_index]
            sorted_D = D[sorted_index]
            
            F = np.sum(sorted_D[sorted_y==1])
            
            if F > 0.5:
                F_min = 1 - F
                b = -1
            else:
                F_min = F
                b = 1
                
            if F_min < F_star:
                F_star = F_min
                self.theta = sorted_val[0] - 1
                self.j = col
                self.b = b
                
            for i in range(nrow-1):
                
                F = F - sorted_y[i]*sorted_D[i]
                
                if F > 0.5:
                    F_min = 1 - F
                    b = -1
                else:
                    F_min = F
                    b = 1
                
                if F_min < F_star and sorted_val[i] != sorted_val[i+1]:
                    F_star = F_min
                    self.theta = 0.5*(sorted_val[i] + sorted_val[i+1])
                    self.j = col
                    self.b = b
                
    def predict(self, X):
        return np.sign((self.theta - X[:, self.j])*self.b)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        
        return accuracy
