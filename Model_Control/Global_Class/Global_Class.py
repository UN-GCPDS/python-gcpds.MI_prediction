## CARGAMOS LOS MODELOS DE EEG

from Model_Control.Models.DeepConvNet import DeepConvNet
from Model_Control.Models.DMTL_BCI import DMTL_BCI
from Model_Control.Models.EEGNet import EEGNet
from Model_Control.Models.EEGNet_fusion import EEGNet_fusion
from Model_Control.Models.MIN2NET import MIN2NET
from Model_Control.Models.MTVAE_standard import MTVAE
from Model_Control.Models.MTVAE_with_loss import MTVAE_KL
from Model_Control.Models.PST_attention import PST_attention
from Model_Control.Models.ShallowConvNet import ShallowConvNet
from Model_Control.Models.Shallownet_1conv2d import Shallownet_1conv2d
from Model_Control.Models.Shallownet_1conv2d_rff import Shallownet_1conv2d_rff
from Model_Control.Utils.Load_datasets import load_dataset
from Model_Control.Utils.compile_model import getOptimizer
from Model_Control.Utils.compile_model import get_callbacks
from Model_Control.Utils.compile_model import get_loss
from Model_Control.Utils.training_model import train_model_cv



class DatasetControl:
     
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
            """
            ###LOAD DATASET QUEDA ALMACENADO EN VARIABLES DE OBJETO ## INCLUYENDO EL PREPROCESO
            X,Y,sfreq = load_dataset(dataset_name=self.DatasetName,Preprocess=Preprocess,subject_id=subject,low_cut_hz= low_cut_hz,high_cut_hz=high_cut_hz, trial_start_offset_seconds= trial_start_offset_seconds,trial_stop_offset_seconds=trial_stop_offset_seconds,Sessions_Runs = Sessions_Runs)
            
            return X,Y,sfreq



class ModelControl(DatasetControl): 

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

     def __init__(self,Model:str,parameters:dict,DatasetName:str='BCI2A'):
         """
         Parameters
         ----------
         Model : str
             Name of the model to use.
         parameters : dict
             Dictionary with all the parameters to create the model
         """
         ## INICIALIZAMOS CLASE PADRE
         super().__init__(DatasetName)
         self.Model = self.Models[Model](**parameters) ## CONSTRUIMOS EL MODELO CON SUS RESPECTIVOS PARAMETROS
         self.name_model = Model ## HERE ALL GOOD
    
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
     
     def getOptimizer(self,optimizer:str,lr:float = 0.01):
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


     def generateHyperParametersModel(self,optimizer:str = 'adam',lr:float = 0.01, metrics:list(str)=['accuracy'] ,callbacks_names = None,call_args =None,loss_list:list=None,loss_weights:list=[2.5,1]):
         """
         Function to define the hyperparameters and loss functions ## 
         ### optional function if i want to compile with the properties of this library

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

         self.opt = getOptimizer(optimizer)(learning_rate = lr) ## OBTENEMOS EL OPTIMIZADOR
         self.metrics = metrics
         self.loss_weights = loss_weights
            
         if (callbacks_names == None or call_args == None):
                print("=========================================")
                print("====NO SE HA DEFINIDO NINGUN CALLBACK====")
                print("=========================================")
         else:
                self.callbacks = get_callbacks(callbacks_names=callbacks_names,call_args=call_args)
            

         self.loss = get_loss(loss_list)



     def train_model(self,Model = None,optimizer = None,loss = None,callbacks = None,loss_weights = None,X_train=None,Y_train=None,x_val=None,y_val=None):
         
         """
         FUNCTION TO TRAIN A COMPILE MODEL
         -----------------------------------------------
         """
         


         pass

            
            
            
            
            
