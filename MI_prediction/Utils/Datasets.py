import numpy as np

from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import preprocess
from braindecode.preprocessing.windowers import create_windows_from_events

from ..Datasets.Moabb import MOABBDataset_Rest
from .Windowers import create_windows_from_events as create_windows_from_events_rest

def get_labels(X,y, labels):
    idx_l = np.zeros(y.shape,dtype=bool)
    for l in labels:
        idx_l += (y==l)
    X = X[idx_l,::]
    y = y[idx_l]
    return X, y

def get_epochs(dset, labels=True):
    if labels:
        y = dset[range(len(dset))][1]
    else:
        y = None
    X = dset[range(len(dset))][0]
    return X,y

def get_epochs_by_labels(X,y):
    labels = np.unique(y)
    Xl = {}
    yl = {}
    for i in labels:
        Xl[str(i)] = X[y==i]
        yl[str(i)] = y[y==i]
    return Xl,yl

def get_runs(dset, n_trials=20):
    X,y = get_epochs(dset)
    X,y = get_epochs_by_labels(X,y)

    d_shape = X['0'].shape[0]

    Xr = {}
    yr = {}

    idx = 0
    run = 0
    while idx+n_trials <= d_shape:
        if idx+2*n_trials > d_shape:
            l_idx = d_shape
        else:
            l_idx = idx + n_trials
        
        Xr['run_'+str(run)] = np.concatenate([X[str(i)][idx:l_idx,:,:] for i in X.keys()],axis=0)
        yr['run_'+str(run)] = np.concatenate([y[str(i)][idx:l_idx] for i in y.keys()],axis=0)

        idx += n_trials
        run += 1
    return Xr,yr

class DataLoader():
    def __init__(self, dataset_name="BNCI2014001"):
        super(DataLoader, self).__init__()
        self.dataset_name = dataset_name
        self.dataset = None

    def load_data(self,subject_ids=None):
        self.dataset = MOABBDataset(dataset_name=self.dataset_name, subject_ids=subject_ids)
    
    def preprocess_data(self,preprocessors=None):
            preprocess(self.dataset, preprocessors)
    
    def get_fs(self):
        return self.dataset.datasets[0].raw.info["sfreq"]
    
    def get_trials(self, start_offset, end_offset):
        sfreq = self.get_fs()
        if len(start_offset) == len(end_offset):
            self.trials = {}
            for i in range(len(start_offset)):
                self.trials['win_'+str(i)] = create_windows_from_events(self.dataset,trial_start_offset_samples=int(start_offset[i]*sfreq),
                    trial_stop_offset_samples=int(end_offset[i]*sfreq), preload=True)
        return self.trials

class DataLoader_Rest():
    def __init__(self, dataset_name):
        super(DataLoader_Rest, self).__init__()
        self.dataset_name = dataset_name

    def load_data(self,subject_ids=None):
        self.dataset = MOABBDataset_Rest(dataset=self.dataset_name, subject_ids=subject_ids)
    
    def preprocess_data(self,preprocessors=None):
            preprocess(self.dataset, preprocessors)
    
    def get_fs(self):
        return self.dataset.datasets[0].raw.info["sfreq"]
    
    def get_trials(self, start_offset, end_offset):
        sfreq = self.get_fs()
        if len(start_offset) == len(end_offset):
            self.trials = {}
            for i in range(len(start_offset)):
                self.trials['win_'+str(i)] = create_windows_from_events_rest(self.dataset,trial_start_offset_samples=int(start_offset[i]*sfreq),
                    trial_stop_offset_samples=int(end_offset[i]*sfreq), preload=True)
        return self.trials