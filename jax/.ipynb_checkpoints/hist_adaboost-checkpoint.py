import jax.numpy as jnp
from jax import jit, vmap

def discretize(x):
    """
    discretize input data into bins
    """
    min_ = x.min()
    max_ = x.max() + 0.050
    bins = jnp.linspace(min_, max_, 256)
    x_binned = jnp.digitize(x, bins, right=False)
    return x_binned, bins

jit_batched_discretize = jit(vmap(discretize, in_axes=(1)))

def split_bincount(x, y, d, F):
    """
    find the best split according to D using an histogram of bins
    """
    bincount = jnp.bincount(x, weights=-1*d*y, length=256)
    obj = F+bincount.cumsum()
    obj = jnp.abs(obj-0.5)
    argmax = obj.argmax()

    return obj[argmax], argmax

jit_batched_split_bincount = jit(vmap(split_bincount, in_axes=(0, None, None, None)))

def predict_(x, j, theta):
      return jnp.sign(theta - x.at[j].get())

jit_predict = jit(vmap(predict_, in_axes=(0, None, None)))

class HistDecisionStump:
    """
    Decision Stump classifier, best split found using an histograms of bins
    """

    def fit(self, x, y, bins, d):
        """
        x is a biined data matrix of dimension 256 x #data
        bins is a matrix of bin edges of dimension 256 x #features
        """

        F = d[y==1].sum()

        objs, bin_indices = jit_batched_split_bincount(x_binned, y, d, F)
        j = objs.argmax()
        bin_index = bin_indices[j]
        theta = bins[j, bin_index+1]
        self.j, self.theta = j, theta

        return self

    def predict(self, x):
        return jit_predict(x, self.j, self.theta)

    def score(self, x, y):
        y_pred = self.predict(x)
        score = (y==y_pred).mean()
        return score
    
@jit
def update_d(d, w, L):
    d = d*(jnp.exp(-1*w*L))
    d = d/d.sum()
    return d  

class HistAdaBoost:

    def __init__(self, T=100):
        self.T = T
        self.estimators = []
        self.ws = []
        self.errors_ = []
    
    def set_params(self, **parameters):
        self.estimator.set_params(**parameters)
        return self
        
    def fit(self, x, y):
        nrow, ncol = x.shape
        d = jnp.full(nrow, 1/nrow)

        # decision stump preprocessing
        x_binned, bins = jit_batched_discretize(x)

        for t in range(self.T):
            estimator = HistDecisionStump()
            estimator.fit(x_binned, y, bins, d)

            y_pred = estimator.predict(x)
            L = y_pred*y

            e = d[L <= 0].sum()


            if abs(e) <= 1e-5 or abs(e-1) <= 1e-5:
                self.estimators = [estimator]
                self.ws = [1]
                return self

            w = 0.5*jnp.log((1-e)/e)
            d = update_d(d, w, L)

            self.estimators.append(estimator)
            self.ws.append(w)
            self.errors_.append(e)

        return self

    def decision_function(self, x):
        y_pred = jnp.zeros(x.shape[0])
        for w, estimator in zip(self.ws, self.estimators):
            y_pred = y_pred + w*estimator.predict(x)

        return y_pred

    def predict(self, x):
        margin = self.decision_function(x)
        y_pred = jnp.sign(margin)

        return y_pred

    def score(self, x, y):
        y_pred = self.predict(x)
        score = (y==y_pred).mean()
        return score
    
    def clone(self):
        return HistAdaBoost(self.T)