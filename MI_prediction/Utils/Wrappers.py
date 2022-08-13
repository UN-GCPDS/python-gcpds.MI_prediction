from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import copy
import numpy as np

class MultiInputScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, input_feats, keep_dims=False):
        self.scaler = scaler
        self.input_feats = input_feats
        self.keep_dims = keep_dims
    
    def get_index(self):
        id = 0 
        idx = []
        for i in range(len(self.input_feats)):
            idx.append((id,id+self.input_feats[i]))
            id+=self.input_feats[i]
        return idx
    
    def feature_encoder(self):
        idx = self.get_index()
        return FunctionTransformer(
            func=lambda X: [X[:, i[0]:i[1]] for i in idx])

    def fit(self, X):
        X = self.feature_encoder().transform(X)
        self.scalers = [copy.deepcopy(self.scaler).fit(x) for x in X]
        return self

    def transform(self, X):
        X = self.feature_encoder().transform(X)
        Xt = [self.scalers[i].transform(X[i]) for i in range(len(X))]
        if self.keep_dims:
            Xt = np.concatenate(Xt, axis=-1)
        return Xt


class MultiInputRegressor(KerasRegressor):

    def get_index(self):
        id = 0 
        idx = []
        for i in range(len(self.input_feats)):
            idx.append((id,id+self.input_feats[i]))
            id+=self.input_feats[i]
        return idx
    
    @property
    def feature_encoder(self):
        idx = self.get_index()
        return FunctionTransformer(
            func=lambda X: [X[:, i[0]:i[1]] for i in idx],
        )