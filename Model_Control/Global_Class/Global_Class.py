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
             - PHYSIONET
             Represent the dataset to use, by default 'BCI2A'
         """
         
         self.DatasetName = self.datasets[DatasetName]
         self.X_train = None
         self.y_train = None
         self.X_valid = None
         self.y_valid = None
         self.sfreq = None
         self.info = None
         self.callbacks = None
         self.metrics = None
    
     def LoadDataset(self,Preprocess:list=None,subject:int = 1,low_cut_hz:float = 4,high_cut_hz:float = 38,trial_start_offset_seconds:float = -0.5,trial_stop_offset_seconds:float = 0):
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
            self.X_train,self.y_train,self.X_valid,self.y_valid,self.sfreq,self.info = load_dataset(dataset_name=self.DatasetName,Preprocess=Preprocess,subject_id=subject,low_cut_hz= low_cut_hz,high_cut_hz=high_cut_hz, trial_start_offset_seconds= trial_start_offset_seconds,trial_stop_offset_seconds=trial_stop_offset_seconds)
         


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
         ## IN ICIALIZAMOS CLASE PADRE
         super().__init__(DatasetName)
         self.Model = self.Models[Model](**parameters) ## CONSTRUIMOS EL MODELO CON SUS RESPECTIVOS PARAMETROS
         self.name_model = Model
          
     def CompileModel(self,optimizer:str = 'adam',lr:float = 0.01, metrics:list(str)=['accuracy'] ,callbacks_names = None,call_args =None):
         """
         Function to define the hyperparameters and loss functions

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

         
         if (self.X_train == None):
            print("===========================================")
            print("====NO HA SIDO CARGADO LA BASE DE DATOS====")
            print("=USA EL METODO LoadDataset PARA PODER OBTENER LA INFORMACIÓN DEL SUJETO=")
         else:
            self.opt = getOptimizer(optimizer)(learning_rate = lr) ## OBTENEMOS EL OPTIMIZADOR
            self.metrics = metrics
            
            if (callbacks_names == None or call_args == None):
                print("=========================================")
                print("====NO SE HA DEFINIDO NINGUN CALLBACK====")
                print("=========================================")
            else:
                self.callbacks = get_callbacks(callbacks_names=callbacks_names,call_args=call_args)
            
            ### PENSAR COMO PERMITIR AGREGAR FUNCIONES CUSTOM
            self.loss = get_loss(self.name_model)


            ### COMPILAMOS EL MODELO

            self.Model.compile(loss=self.loss, optimizer= self.opt, metrics=self.metrics, loss_weights=self.loss_weights)
     

     def train_model(self):
         pass

            
            
            
            
            
