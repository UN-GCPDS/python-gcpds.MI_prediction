from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing.windowers import create_windows_from_events
import numpy as np
import tensorflow as tf 

def name_to_numclasses(class_names):

    """
    class_names:list of name of each class
    Returns
    -------
    list  of  ordinal encoding for each class
    """

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

def getSessionsRuns(datasetName,sbj_id:int = None):
         

         runs = {
         'BNCI2014001':['0', '1', '2', '3', '4', '5'],
         'Cho2017':['0','1','2','3','4'],
         }
         sessions = {
            'BNCI2014001':['1test', '0train'],
            'Cho2017':['0']
         }
         
         runs_7 = {
         'BNCI2014001':['0', '1', '2', '3', '4', '5'],
         'Cho2017':['0','1','2','3','4','5'],
         }
         if (sbj_id != None):
            return {'sessions':sessions[datasetName],'runs':runs_7[datasetName]}

         return {'sessions':sessions[datasetName],'runs':runs[datasetName]}


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


def load_dataset(dataset_name:str="BNCI2014001", subject_id:int=1, low_cut_hz:float = 4., high_cut_hz:float = 38., trial_start_offset_seconds:float = -0.5,trial_stop_offset_seconds:float=0,Preprocess=None,Sessions_Runs:dict = None):
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
    
    Sessions_Runs: dict , optional , default value None in that case you will load all the database for that subject
                  dictionary with two keys [sessions,runs] 
                  -sessions : list with the names of the sessions that you want to load
                  -runs : list with the names of the runs that you want to load, 
                  you can verify all that names, with the function getSessionsRuns that its related with the dataset that you selectionate when you create the object

    Returns
    -------
    X_data => [[session[runs]]]  runs => (trial,channels,time_serie)
    y_data => [[session[runs]]]  runs => (labels,)
    sfreq   => frequency of sampling  float
    """


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

    if (Sessions_Runs == None):

        ### EN ESTE CASO OBTENEMOS TODAS LAS SESSIONES Y TODOS LOS RUNS

        splitted = windows_dataset.split('session')
        session_run = getSessionsRuns(dataset_name)
        sessions = session_run['sessions']
        
        X = []
        y = []
        for sesion in sessions:

            X_session,y_session = get_epochs(splitted[sesion])
            X.append(X_session)
            y.append(y_session)
        
        ## DEVOLVEMOS UNA LISTA SEGMENTADA POR SESION CON TODOS LOS RUNS

        return np.array(X),np.array(y),sfreq
    

    else: 
        if(dataset_name == 'Cho2017'):
           
           runs_index={
             '0':[0,20,100,120],
             '1':[20,40,120,140],
             '2':[40,60,140,160],
             '3':[60,80,160,180],
             '4':[80,100,180,200]
           }
           runs_index_7={
             '0':[0,20,120,140],
             '1':[20,40,140,160],
             '2':[40,60,160,180],
             '3':[60,80,200,220],
             '4':[80,100,220,240],
             '5':[100,120,240,260],
           }
           if(subject_id == 7 or subject_id == 9):
                ### CON GIGA NECESITAMOS UN PROCESO DIFERENTE
                ### PRIMERO CARGAMOS TODA LA BASE DE DATOS
                splitted = windows_dataset.split('session')
                session_run = getSessionsRuns(dataset_name,sbj_id=subject_id)
                sesion = session_run['sessions'][0]
                runs = Sessions_Runs['runs']
                X_session,y_session = get_epochs(splitted[sesion])
                X_s = []
                y_s = []
                X = []
                y = []
                for run in runs:
                    
                    index_run = runs_index_7[run]
                    ### VERIFICAR ESTA BASE DE DATOS
                    X_run_0 = X_session[index_run[0]:index_run[1],:,:,:]
                    X_run_1 = X_session[index_run[2]:index_run[3],:,:,:]
                    X_run = np.concatenate((X_run_0,X_run_1),axis = 0) 

                    y_run_0 = y_session[index_run[0]:index_run[1]]
                    y_run_1 = y_session[index_run[2]:index_run[3]]
                    y_run = np.concatenate((y_run_0,y_run_1),axis = 0)
                    
                    ### run por sesion
                    X_s.append(X_run)
                    y_s.append(y_run)
                

                X.append(X_s)
                y.append(y_s)

                return np.array(X),np.array(y),sfreq
           else:
               
                ### CON GIGA NECESITAMOS UN PROCESO DIFERENTE
                ### PRIMERO CARGAMOS TODA LA BASE DE DATOS
                splitted = windows_dataset.split('session')
                session_run = getSessionsRuns(dataset_name)
                sesion = session_run['sessions'][0]
                runs = Sessions_Runs['runs']
                X_session,y_session = get_epochs(splitted[sesion])
                X_s = []
                y_s = []
                X = []
                y = []
                for run in runs:
                    
                    index_run = runs_index[run]
                    ### VERIFICAR ESTA BASE DE DATOS
                    X_run_0 = X_session[index_run[0]:index_run[1],:,:,:]
                    X_run_1 = X_session[index_run[2]:index_run[3],:,:,:]
                    X_run = np.concatenate((X_run_0,X_run_1),axis = 0) 

                    y_run_0 = y_session[index_run[0]:index_run[1]]
                    y_run_1 = y_session[index_run[2]:index_run[3]]
                    y_run = np.concatenate((y_run_0,y_run_1),axis = 0)
                    
                    ### run por sesion
                    X_s.append(X_run)
                    y_s.append(y_run)
                

                X.append(X_s)
                y.append(y_s)

                return np.array(X),np.array(y),sfreq

           


        else:
            sesiones_objetivo = Sessions_Runs['sessions']
            runs_objetivo = Sessions_Runs['runs']

            ### PRIMERO OBTENEMOS UN DICTIONARIO SEPARADO POR SESSIONES
            splitted = windows_dataset.split('session')
            ### POR CADA SESSION SEPARAMOS POR RUNS
            dictionary_dataset = {}
            for sesion in sesiones_objetivo:
                list_runs = []
                dataset_runs = splitted[sesion].split('run')
                for run in runs_objetivo:
                    list_runs.append(dataset_runs[run]) ## AGREGAMOS EL RUN CORRESPONDIENTE
                
                dictionary_dataset[sesion] = list_runs ## GUARDAMOS LA LISTA CON CADA RUN
            
            ### ORGANIZAMOS CADA UNA DE LAS BASES DE DATOS
            X = []
            y = []
            for sesion in sesiones_objetivo:
                x_sesion = []
                y_sesion = []
                for run in range(0,len(dictionary_dataset[sesion])):
                    ## OBTENEMOS LOS DATOS DE CADA RUN SEPARADO YA EN ETIQUETA Y LABEL
                    x_run,y_run = get_epochs(dictionary_dataset[sesion][run])
                    x_sesion.append(x_run)
                    y_sesion.append(y_run)
                X.append(x_sesion)
                y.append(y_sesion)
            
            return np.array(X),np.array(y),sfreq




