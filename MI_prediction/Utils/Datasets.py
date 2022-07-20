from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import preprocess
from braindecode.preprocessing.windowers import create_windows_from_events
import numpy as np

def get_epochs(dset):
    y = dset[range(len(dset))][1]
    X = dset[range(len(dset))][0]
    return X,y

class DataLoader():
    def __init__(self, dataset_name="BNCI2014001"):
        super(DataLoader, self).__init__()
        self.dataset_name = dataset_name

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