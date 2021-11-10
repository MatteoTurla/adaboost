import numpy as np
import abc
from abc import ABC

class BaseClassifier(ABC):
    """
    Base Classifier class. All implemented classifiers inherit from this class
    """

    def set_params(self, **parameters):
        """
        set attributes of the classifier
        used by GridSearchCV to change the attributes of the classifier to tune
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    @abc.abstractmethod 
    def preprocessing(self, X):
        """
        Preprocessing steps. 
        this function output a dictionary that must be passed to the fit method
        
        Usefull to speed-up computation when the classifier is called multiple times on the same data matrix X

        Parameters
        ----------
        X: m by d matrix
            matrix of examples
        
        Output
        ------
        dictionary: contains information used by the fit method
        """
        return
    
    @abc.abstractmethod 
    def fit(self, X, y, D, kwargs):
        """
        Learn the classifier

        Parameters
        ----------
        X: m by d matrix
            matrix of examples
        y: m by k matrix
            matrix of labels.
            in case of binary classifier it is a mx1 vector
            in case of multiclass classifier it is a mxK matrix, where K is number of unique labels
        D: m by 1 vector of weights
        kwargs: optional dict aregument specific for each classifier
        
        Output
        ------
        self: learned classifier 
        """
        return
    
    @abc.abstractmethod
    def predict(self, X):
        """
        Compute prediction over X
        
        Parameters
        ----------
        X: m by d matrix
        
        Outputs
        -------
        y_pred: m by 1 vector
        """
        return
        
    @abc.abstractmethod
    def decision_function(self, X):
        """
        Compute a 'confidence-value' for each prediction
        
        If the classifier use as prediction rule: sign(F(X)) then decision function return f(X)
        
        Parameters
        ----------
        X: m by d matrix
        
        Outputs
        -------
        y_pred: m by 1 vector
        """
        return
        
    def score(self, X, y):
        """
        Compute accuarcy: 1 - zero-one loss
        """
        y_pred = self.predict(X)
        score = (y_pred == y).mean()
        return score
    
    @abc.abstractmethod
    def clone(self):
        """
        Return a new instance of the current class
        """
        return