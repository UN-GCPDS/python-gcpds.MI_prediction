from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import FunctionTransformer

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