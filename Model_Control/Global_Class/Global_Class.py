## CARGAMOS LOS MODELOS DE EEG
import tensorflow as tf
from ..Models.DeepConvNet import DeepConvNet
from ..Models.DMTL_BCI import DMTL_BCI
from ..Models.EEGNet import EEGNet
from ..Models.EEGNet_fusion import EEGNet_fusion
from ..Models.MIN2NET import MIN2NET
from ..Models.MTVAE_standard import MTVAE
from ..Models.MTVAE_with_loss import MTVAE_KL
from ..Models.PST_attention import PST_attention
from ..Models.ShallowConvNet import ShallowConvNet
from ..Models.Shallownet_1conv2d import Shallownet_1conv2d
from ..Models.Shallownet_1conv2d_rff import Shallownet_1conv2d_rff
from ..Utils.Load_datasets import load_dataset
from ..Utils.compile_model import getOptimizer
from ..Utils.compile_model import get_callbacks
from ..Utils.compile_model import get_loss
from ..Utils.training_model import redirectToTrain



class DatasetControl():
     
     """
     Class to control de loading of eeg database
     -----------------------------------------------
     Databases 
     - BCI2A
     - GIGA
     """
     
     datasets = {
         'GIGA':'Cho2017',
         'BCI2A':'BNCI2014001',
     }
    
     def __init__(self,DatasetName:str='BCI2A'): 
         """
         Parameters
         ----------
         DatasetName : str,
             - BCI2A 
             - GIGA
             Represent the dataset to use, by default 'BCI2A'
         """
         
         self.DatasetName = self.datasets[DatasetName]

     
     def getSessionsRuns(self):
         
         """
         Function to return the segmentation of database

         Returns
         -------
         dictionary:
            -sessions: represent an array with the name of each session in the database
            -runs : represent an array with the name of each run in the database
         """
         

         runs = {
         'BNCI2014001':['run_0', 'run_1', 'run_2', 'run_3', 'run_4', 'run_5'],
         'Cho2017':['run_0','run_1','run_2','run_3','run_4'],
         }
         sessions = {
            'BNCI2014001':['session_E', 'session_T'],
            'Cho2017':['session_0']
         }


         return {'sessions':sessions[self.DatasetName],'runs':runs[self.DatasetName]}



     def LoadDataset(self,Preprocess:list=None,subject:int = 1,low_cut_hz:float = 4,high_cut_hz:float = 38,trial_start_offset_seconds:float = -0.5,trial_stop_offset_seconds:float = 0, Sessions_Runs:dict = None):
            """
            Parameters
            ----------
            subject : int, optional
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
            
            Return
            -----------------------------------

            X (list): list that it's divided with a array of each session and run  time series (trials,Channels,time_serie)
            Y (list): list that it's divided with a array of each session and run label or class
            sfreq : float with the sample frequency
            """
            ###LOAD DATASET QUEDA ALMACENADO EN VARIABLES DE OBJETO ## INCLUYENDO EL PREPROCESO
            X,Y,sfreq = load_dataset(dataset_name=self.DatasetName,Preprocess=Preprocess,subject_id=subject,low_cut_hz= low_cut_hz,high_cut_hz=high_cut_hz, trial_start_offset_seconds= trial_start_offset_seconds,trial_stop_offset_seconds=trial_stop_offset_seconds,Sessions_Runs = Sessions_Runs)
            
            return X,Y,sfreq



class ModelControl(DatasetControl): 
     
     """
     Model to control the training of a eeg model you can:

      -Load  a dataset of MI [BCI2A,GIGA]
      -Load a specific model [DeepConvNet,DMTL_BCI,EEGNet,MIN2NET,MTVAE_standard,MTVAE_with_loss,MTVAE_with_loss,PST_attention,ShallowConvNet,Shallownet_1conv2d,Shallownet_1conv2d_rff]
      -Organize callbacks with tensorflow
      -Organize optimizer of tensorflow
      -Train model with diferent validation mode
     """

     Models = {
          'DeepConvNet': DeepConvNet,
          'DMTL_BCI': DMTL_BCI,
          'EEGNet': EEGNet,
          'EEGNet_fusion':EEGNet_fusion,
          'MIN2NET':MIN2NET,
          'MTVAE_standard':MTVAE,
          'MTVAE_with_loss':MTVAE_KL,
          'PST_attention':PST_attention,
          'ShallowConvNet':ShallowConvNet,
          'Shallownet_1conv2d':Shallownet_1conv2d,
          'Shallownet_1conv2d_rff':Shallownet_1conv2d_rff,
     }

     def __init__(self,Model:str = None,parameters:dict = None,DatasetName:str='BCI2A'):
         """
         Parameters
         ----------
         Model : str
             Name of the model to use. (necessary)
         parameters : dict
             Dictionary with all the parameters to create the model (necessary)
         """
         ## INICIALIZAMOS CLASE PADRE
         super().__init__(DatasetName)
         if(Model  == None):
            self.Model = None ## DEJAMOS QUE UTILIZEN SU PROPIO MODELO
            self.name_model = None
         else:
            self.Model = self.Models[Model](**parameters) ## CONSTRUIMOS EL MODELO CON SUS RESPECTIVOS PARAMETROS
            self.name_model = Model ## HERE ALL GOOD
            self.callbacks = None
            self.opt = None
            self.metrics = None
            self.loss = None
            self.loss_weights = [2.5,1]
    
     def getCallBack(self,callbacks_names,call_args):
         """
         Parameters
         -------------------------
         callbacks_names (dict)
         call_args (list)
         """
         callbacks = get_callbacks(callbacks_names,call_args)
         return callbacks
     
     def getLoss(self,loss_list:list):
          """
          parameters
          ------------------------------
          loss_list list(str): list to get loss functions
             options :
                -mse
                -msle
                -binary_crossentropy
                -binary_focal_crossentropy
                -categorical_crossentropy
                -categorical_focal_crossentropy
                -CategoricalHinge
                -CosineSimilarity
                -Hinge
                -Huber
                -LogCosh
                -Poisson
                -SparseCategoricalCrossentropy
                -SquaredHinge
        
          return 
          -------------------------
          list [tf.keras.losses.functions]

          """
          losses = get_loss(loss_list)

          return losses
     
     def get_Optimizer(self,optimizer:str,lr:float = 0.01):
          """
          parameters
          ----------------------------
          optimizer (str) = Name of optimizer to use
              - Adadelta
              - Adafactor
              - Adagrad
              - Adam
              - AdamW
              - Adamax
              - Ftrl
              - Lion
              - Nadam
              - RMSprop
              - SGD
          lr (float) = learning rate for optimizer default value (0.01)

          return 
          -------------------------
          optimizer [tf.keras.optimizer]
          """
          opt = getOptimizer(optimizer)(learning_rate = lr)
          return opt


     def compileModel(self,Model = None,optimizer:str = 'adam',lr:float = 0.01, metrics:list(str)=['accuracy'] ,loss_list:list=None,loss_weights:list=[2.5,1]):
         """
         Function to define the hyperparameters and loss functions ## 
         ### optional function.

         Parameters
         -----------------------------
         optimizer (str) = Name of optimizer to use
              - Adadelta
              - Adafactor
              - Adagrad
              - Adam
              - AdamW
              - Adamax
              - Ftrl
              - Lion
              - Nadam
              - RMSprop
              - SGD
         metrics list(str): default ['accuracy']
         """

         if (Model == None):
             
             if(self.Model == None):
                 print("======================================")
                 print("NO SE HA DEFINIDO UN MODELO PARA COMPILAR")
                 print("======================================")
             else:
                 ### COMPILAMOS EL MODELO SELECCIONADO
                 self.opt = getOptimizer(optimizer)(learning_rate = lr) ## OBTENEMOS EL OPTIMIZADOR
                 self.metrics = metrics
                 self.loss_weights = loss_weights
                 self.loss = get_loss(loss_list)
                 tf.keras.backend.clear_session()
                 tf.random.set_seed(self.seed)
                 self.Model.compile(loss=self.loss, optimizer= self.optimizer, metrics=self.metrics, loss_weights=self.loss_weights)
                 print("======================================")
                 print("MODELO COMPILADO EXITOSAMENTE")
                 print("======================================")

         else:
             
             ### COMPILAMOS EL MODELO SELECCIONADO
                 self.opt = getOptimizer(optimizer)(learning_rate = lr) ## OBTENEMOS EL OPTIMIZADOR
                 self.metrics = metrics
                 self.loss_weights = loss_weights
                 self.loss = get_loss(loss_list)
                 self.Model = Model
                 tf.keras.backend.clear_session()
                 tf.random.set_seed(self.seed)
                 self.Model.compile(loss=self.loss, optimizer= self.optimizer, metrics=self.metrics, loss_weights=self.loss_weights)
                 print("======================================")
                 print("MODELO COMPILADO EXITOSAMENTE")
                 print("======================================")
        
     

     def train_model(self,Model = None,X_train=None,Y_train=None,x_val=None,y_val=None,callbacks_names = None,call_args = None,validation_mode:str = None, batch_size:int =30,epochs:int = 100,verbose:int =1):
         
         """
        Parameters
        ------------------------------------------
        Model : tensorFlowModel
            model of tensorFlow compiled
        callbacks : 
            callbacks for train the model defined by the function of get_callbacks
        X_train : array
            input training array data
        Y_train : array
            target training array data
        x_val : array
            input validation array data
        y_val : array
            target validation array data
        
        validation_mode:str default None : 
            strategy of validation if validation_mode is None, the validation strategy is the conventional just comparing
            the training and validation data during one training. 
        batch_size : int 
            segmentation of the training data during training.
        epochs : int 
            number of epochs to train the model
        verbose : int [0,1]
            option during training to watch the description of the training. 0 didn't print anything and 1 print all the information
        
        Return
        ---------------------------------------------------------
        History: could be a list or a object with the information of each training for the selected model
        X_val : input data to evaluate the model
        y_val : target data for evaluate the model
        """
         
         if (call_args == None or callbacks_names == None):
             print("=============================================================================================================")
             print("Deben suministrarse los argumentos de callbacks_names y call_args para obtener los callbacks correspondientes")
             print("el modelo se entrenara sin ningun callback")
             print("=============================================================================================================")
             self.callbacks = None
         else:
             ###OBTENEMOS LOS CALLBACKS SELECCIONADOS
             self.callbacks = get_callbacks(callbacks_names,call_args)

         if(Model == None):
            ### VERIFICAMOS EL MODELO PROPIO PARA SABER SI SE COMPILO

            if(self.Model.compiled):
                
                self.Model, History , x_val , y_val=redirectToTrain(self.Model,self.callbacks,X_train,Y_train,x_val,y_val,validation_mode, batch_size,epochs,verbose)
                
                ## PARA CALCULAR EL ACCURRACY UNA VEZ LO TENGA CLARO HASTA ESTE PUNTO PROCEDEMOS A GENERAR ESE APARTADO
                return History,x_val,y_val
                
            else:
               
               print("==============================================================================")
               print("NO SE HA SUMINISTRADO UN MODELO Y EL MODELO DE LA CLASE AUN NO SE HA COMPILADO")
               print("==============================================================================")
               return None,None,None

            
         else:
            
            if(Model.compiled):
                self.Model = Model ### DEFINIMOS EL MODELO COMO PROPIO DEL OBJETO
                self.Model, History , x_val , y_val=redirectToTrain(self.Model,self.callbacks,X_train,Y_train,x_val,y_val,validation_mode, batch_size,epochs,verbose)
                return History,x_val,y_val
            else:
               
               print("==============================================================================")
               print("NO SE HA SUMINISTRADO UN MODELO COMPILADO")
               print("==============================================================================")
               return None,None,None
            
            
            
            
            
