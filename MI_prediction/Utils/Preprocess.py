from braindecode.preprocessing.preprocess import Preprocessor
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
import numpy as np
import copy

def moments(epoch, axis=0):
    return np.concatenate([epoch[axis].mean(axis=-1,keepdims=True),epoch[axis].var(axis=-1,keepdims=True),
    epoch[axis].max(axis=-1,keepdims=True),epoch[axis].min(axis=-1,keepdims=True),
    np.median(epoch[axis],axis=-1,keepdims=True)],axis=-1)

def filterbank_preprocessor(freqs):
    FB = {}
    for freq in freqs:
        print('band to filter: {} Hz'.format(freq))
        FB[str(freq[0])+'_'+str(freq[1])] = Preprocessor('filter', l_freq=freq[0], h_freq=freq[1])
    return FB

def filterbank(ds, preprocess = [], filters = [], standarization = [], channels_prep = []):
    ds_filt = []
    for i in filters.keys():
        d_tmp = copy.deepcopy(ds)
        preprocessors = preprocess + channels_prep + [filters[i]] + standarization
        d_tmp.preprocess_data(preprocessors=preprocessors)
        ds_filt.append(d_tmp)
    return ds_filt

class FBCSP(TransformerMixin,BaseEstimator):
    def __init__(self, n_components=8, reg=None, log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
    
    def fit(self, X, y):
        self.csp_mdl = CSP(n_components=self.n_components, reg=self.reg, log=self.log, norm_trace=self.norm_trace)
        self.mdls = [self.csp_mdl.fit(X[:,:,:,i], y) for i in range(X.shape[-1])]
        return self
    
    def transform(self, X):
        return np.concatenate([self.mdls[i].transform(X[:,:,:,i]) for i in range(len(self.mdls))],axis=-1)
