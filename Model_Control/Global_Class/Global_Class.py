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

#### CARGAMOS LAS BASES DE DATOS



class ModelControl: 

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

     def __init__(self,Model:str,parameters:dict):
         """
         Parameters
         ----------
         Model : str
             Name of the model to use.
         parameters : dict
             Dictionary with all the parameters to create the model
         """
         self.Model = self.Models[Model](**parameters)
          
     def compileModel(self):
        pass


class DatasetControl:
     
     datasets = {
         'GIGA':'Cho2017',
         'BCI2A':'BNCI2014001',
     }
    
     def __init__(self,DatasetName:str,Preprocess:dict,subjects:list): 
         """
         Parameters
         ----------
         DatasetName : str
             - name of the datasets [GIGA,BCI2A]
         Preprocess : dict
             - dictionary with all the preprocess data

         subjects : list 
            - list with the id of the subject that you want to load the data 
         """
         

