from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing.windowers import create_windows_from_events
import numpy as np
import tensorflow as tf 

def name_to_numclasses(class_names):

    classes = []
    for i in class_names:
        if i=='left hand':
            classes.append(0)
        elif i=='right hand':
            classes.append(1)
        elif i=='feet':
            classes.append(2)
        elif i=='tongue':
            classes.append(3)
    return classes


def get_classes(X,y, class_names):
    classes = name_to_numclasses(class_names)
    X_c = []
    y_c = []
    for i in classes:
        X_c.append(X[y==i,:,:,:])
        y_c.append(y[y==i])
    X_c = np.concatenate(X_c,axis=0)
    y_c = np.concatenate(y_c,axis=0)
    return X_c, y_c


def get_epochs(dset):
    """
    function to organize data in to sets (trials,channels,time_serie) and (labels)

    Parameters
    ----------
    dset : Array of data 
    Returns
    -------
    X (array) => (trials,channels,time_serie) 
    y (array) => (labels)

    """
    y = []
    X = []
    for i in range(len(dset)):
        y.append(dset[i][1])
        X.append(np.expand_dims(dset[i][0],axis=[0,3]))
    
    y = np.asarray(y)
    X = np.concatenate(X,axis=0)
    return X,y

def getChannels(dataset_name:str):
    """
    Parameters
    ----------
    dataset_name : str
        [Cho2017,BNCI2014001]
    """
    ##JUST EEG CHANNELS

    DataChannels = {
        'Cho2017': [
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
            "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
            "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ],
        'BNCI2014001': [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"
        ]
    }

    return DataChannels[dataset_name]



def load_dataset(dataset_name:str="BNCI2014001", subject_id:int=1, low_cut_hz:float = 4., high_cut_hz:float = 38., trial_start_offset_seconds:float = -0.5,trial_stop_offset_seconds:float=0,Preprocess=None):
    """
    Parameters
    ----------
    dataset_name : str, optional
        Name dataset to load, by default "BNCI2014001"
    subject_id : int, optional
        Id subject to load, by default 1
    low_cut_hz : float, optional
        low frequency cut, by default 4.
    high_cut_hz : float, optional
        high frequency cut, by default 38.
    trial_start_offset_seconds : float, optional
        , by default -0.5
    trial_stop_offset_seconds : float, optional
        , by default 0
    Preprocess : list , optional of Preprocessor from braindecode.preprocessing.preprocess
    Returns
    -------
    X_train => (trials,channels,time_serie) matriz
    y_train => (trials,channels,time_serie) matriz
    X_valid => (trials,channels,time_serie) matriz
    y_valid => (trials,channels,time_serie) matriz
    sfreq   => frequency of sampling  float
    info    => general information for dataset dataframe
    """

    ### DEFINIMOS SEGMETACIÓN SEGUN LA BASE DE DATOS
    sessions = {
        'BNCI2014001':True,
        'Cho2017':False,
    }
    ##CARGAMOS LA BASE DE DATOS
    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id]) ## CARGAMOS LA BASE DE DATOS

    
    # Parameters for exponential moving standardization
    
    ##Obtenemos los canales de la base de datos
    Channels = getChannels(dataset_name)
    if Preprocess == None:
        factor_new = 1e-3 ##DEFINIR SI PUEDE SER DINAMICO O NO
        init_block_size = 1000 ## DEFINIR SI PUEDE SER DINAMICO O NO
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor('pick_channels',ch_names=Channels),
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
            factor_new=factor_new, init_block_size=init_block_size)
        ]
        # Transform the data
        preprocess(dataset, preprocessors)
    else:
        preprocess(dataset, Preprocess)
    
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=int(trial_stop_offset_seconds*sfreq),
        preload=True,
    )

    splitted = windows_dataset.split('session')


    if sessions[dataset_name]:
        sess1 = 'session_T'
        sess2 = 'session_E'
    else:
        sess1 = 'session_0'
        sess2 = 'session_0'
    
    train_set = splitted[sess1]
    valid_set = splitted[sess2]

    X_train,y_train = get_epochs(train_set)
    X_valid,y_valid = get_epochs(valid_set)

    # if Classes is not None:
    #     X_train,y_train = get_classes(X_train,y_train, Classes)
    #     X_valid,y_valid = get_classes(X_valid,y_valid, Classes)
    info = dataset.datasets[0].raw.info
    return X_train,y_train,X_valid,y_valid,sfreq,info


### COMPILE MODEL


