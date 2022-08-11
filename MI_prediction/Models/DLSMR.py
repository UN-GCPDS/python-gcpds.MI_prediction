import numpy as np
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model



def DLSMR(input_feats = [], Q1f = 1.5, Qff = 0.5, l1 = 1e-3, l2 = 1e-3, act1 = 'tanh', actf = 'tanh', seed = 100):

    winitializer = GlorotNormal(seed=seed)
    binitializer = "zeros"
    
    n_inputs = len(input_feats)

    inputs = n_inputs*[None]
    hiddens = n_inputs*[None]

    for i in range(n_inputs):
        inputs[i] = Input(shape=(input_feats[i]), name='in_'+str(i))
        hiddens[i] = Dense(int(input_feats[i]*Q1f),activation=act1,kernel_regularizer=l1_l2(l1=l1,l2=l2),kernel_initializer=winitializer,bias_initializer=binitializer,name='h1_'+str(i))(inputs[i])
    
    concat = concatenate(hiddens,name='concat')
    Qh = np.sum([input_feats[i]*Q1f for i in range(n_inputs)])
    
    hfi = Dense(Qh*Qff,activation=actf,kernel_regularizer=l1_l2(l1=l1,l2=l2),kernel_initializer=winitializer,bias_initializer=binitializer,name='hfi')(concat)
    
    output_c = Dense(1,activation="linear",kernel_initializer=winitializer, bias_initializer=binitializer,
                                 kernel_regularizer=l1_l2(l1=l1,l2=l2),name='outc')(hfi)
    
    return Model(inputs=inputs,outputs=[output_c])