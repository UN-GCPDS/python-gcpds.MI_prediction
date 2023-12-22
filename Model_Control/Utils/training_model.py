import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split,StratifiedKFold



def get_pred_labels(preds):
        
        pred_labels = np.argmax(preds,axis=-1)
        return pred_labels

    
def get_accuracy(preds,y_true,decimals=2):
    pred_labels = get_pred_labels(preds)
    acc = np.mean(pred_labels==np.argmax(y_true,axis = -1 ))
    return np.round(acc*100,decimals=decimals)


def calAccuracy(Model,X_train,Y_train,x_val,y_val,validation_mode,list_paths,autoencoder):
            if(validation_mode == 'lawhern2018'):

                if(autoencoder):
                    preds = []
                    y_true = []
                    acc = []
                    c = 0

                    skf = StratifiedKFold(n_splits=4)

                    for train_index, test_index in skf.split(X_train, Y_train[1]):

                        ____, X_test_ = X_train[train_index], X_train[test_index]
                        ____, y_test_ = Y_train[1][train_index], Y_train[1][test_index]
                        ### cargamos los pesos
                        Model.load_weights(list_paths[c])
                        pred = Model.predict(X_test_)
                        preds.append(pred[1])
                        y_preds = preds[c].argmax(axis = -1)
                        y_true.append(y_test_)
                        acc.append(np.mean(y_preds == y_test_))
                        print("Fold %d Classification accuracy: %f " % (c+1,acc[c]))
                        c += 1

                    preds = np.concatenate(preds,axis=0)
                    y_true = np.concatenate(y_true,axis=0)
                    acc = get_accuracy(preds,tf.keras.utils.to_categorical(y_true,num_classes=2),decimals=2)
                    return acc
                else:
                    preds = []
                    y_true = []
                    acc = []
                    c = 0

                    skf = StratifiedKFold(n_splits=4)

                    for train_index, test_index in skf.split(X_train, Y_train):

                        ____, X_test_ = X_train[train_index], X_train[test_index]
                        ____, y_test_ = Y_train[train_index], Y_train[test_index]
                        ### cargamos los pesos
                        Model.load_weights(list_paths[c])
                        pred = Model.predict(X_test_)
                        preds.append(pred)
                        y_preds = preds[c].argmax(axis = -1)
                        y_true.append(y_test_)
                        acc.append(np.mean(y_preds == y_test_))
                        print("Fold %d Classification accuracy: %f " % (c+1,acc[c]))
                        c += 1

                    preds = np.concatenate(preds,axis=0)
                    y_true = np.concatenate(y_true,axis=0)
                    acc = get_accuracy(preds,tf.keras.utils.to_categorical(y_true,num_classes=2),decimals=2)
                    return acc
            else:
                 return 'otros métodos de validación no han sido implementados'

def redirectToTrain(Model,callbacks,X_train,Y_train,x_val,y_val,validation_mode, batchSize,epochs,verbose,seed = 20200220,autoencoder=False):
        
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
        """
    
        ### ENTRENAMOS EL MODELO
        if (validation_mode == None):
            if(callbacks == None):
                ### ENTRENAMOS DE MANERA ESTANDAR
                        
                history = Model.fit(X_train,Y_train,validation_data=(x_val,y_val),batch_size=batchSize,epochs=epochs,verbose=verbose)
                preds = Model.predict(x_val)
                if(autoencoder):
                    acc = get_accuracy(preds[1],y_val[1],decimals=2)
                else:
                    acc = get_accuracy(preds,y_val,decimals=2)
                return Model,history,acc
            else:
                ### ENTRENAMOS DE MANERA ESTANDAR
                        
                history = Model.fit(X_train,Y_train,validation_data=(x_val,y_val),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks = callbacks)
                preds = Model.predict(x_val)
                if(autoencoder):
                    acc = get_accuracy(preds[1],y_val[1],decimals=2)
                else:
                    acc = get_accuracy(preds,y_val,decimals=2)
                return Model,history,acc
            
        else:

            if(callbacks == None):

                print("============================================================================================")
                print("==Para aplicar el metodo de validación es necesario definir los callbacks correspondientes==")
                print("============================================================================================")
                return None
            
            else:
                History = []
                if validation_mode=='schirrmeister2017':
                    if(autoencoder):
                            X_tr, X_ts, y_tr, y_ts = train_test_split(X_train,Y_train[1], test_size=0.2,random_state=seed)
                    else:
                            X_tr, X_ts, y_tr, y_ts = train_test_split(X_train,Y_train, test_size=0.2,random_state=seed)
                    callbacks_names = [callbacks['early_stopping_train'],callbacks['checkpoint_train']]
                    if(autoencoder):
                            history1 = Model.fit(X_tr,[X_tr,y_tr],validation_data=(X_ts,[X_ts,y_ts]),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                    else:
                            history1 = Model.fit(X_tr, y_tr,validation_data=(X_ts, y_ts),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                    History.append(history1)
                    stop_epoch= np.argmin(history1.history['val_loss'])
                    loss_stop = history1.history['loss'][stop_epoch]


                    Model.load_weights(callbacks['checkpoint_train'].filepath)

                    callbacks['Threshold_valid'].threshold = loss_stop
                    callbacks['early_stopping_valid'].patience = (stop_epoch)*2
                    callbacks_names = [callbacks['Threshold_valid'],callbacks['checkpoint_valid'],
                                        callbacks['early_stopping_valid']]

                    if(autoencoder):
                        history2= Model.fit(X_train,[X_train,Y_train[1]],validation_data=(x_val,[x_val,y_val[1]]),batch_size=batchSize,epochs=(stop_epoch+1)*2,verbose=verbose,callbacks=callbacks_names)
                    else:
                        history2= Model.fit(X_train,Y_train,validation_data=(x_val,y_val),batch_size=batchSize,epochs=(stop_epoch+1)*2,verbose=verbose,callbacks=callbacks_names)
                    History.append(history2)
                    Model.load_weights(callbacks['checkpoint_valid'].filepath)

                    preds = Model.predict(x_val)
                    if(autoencoder):
                        acc = get_accuracy(preds[1],y_val[1],decimals=2)
                    else:
                        acc = get_accuracy(preds,y_val,decimals=2)

                    return Model, History , acc


                elif validation_mode=='schirrmeister2017_legal':
                    if(autoencoder):
                        X_tr, X_ts, y_tr, y_ts = train_test_split(X_train,Y_train[1], test_size=0.2,random_state=seed)
                    else:
                        X_tr, X_ts, y_tr, y_ts = train_test_split(X_train,Y_train, test_size=0.2,random_state=seed)
                    
                    callbacks_names = [callbacks['early_stopping_train'],callbacks['checkpoint_train']]
                
                    if(autoencoder):
                        history1 = Model.fit(X_tr, [X_tr,y_tr],validation_data=(X_ts, [X_ts,y_ts]),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                    else:
                        history1 = Model.fit(X_tr, y_tr,validation_data=(X_ts, y_ts),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                    History.append(history1)
                    stop_epoch= np.argmin(history1.history['val_loss'])
                    loss_stop = history1.history['loss'][stop_epoch]

                    Model.load_weights(callbacks['checkpoint_train'].filepath)
                    callbacks['Threshold_valid'].threshold = loss_stop
                    callbacks['early_stopping_valid'].patience = (stop_epoch)*2
                    callbacks_names = [callbacks['Threshold_valid'],callbacks['checkpoint_valid'],callbacks['early_stopping_valid']]
                    
                    if(autoencoder):
                        history2= Model.fit(X_train,[X_train,Y_train[1]],validation_data=(X_ts, [X_ts,y_ts]),batch_size=batchSize,epochs=(stop_epoch+1)*2,verbose=verbose,callbacks=callbacks_names)
                    else:
                        history2= Model.fit(X_train,Y_train,validation_data=(X_ts, y_ts),batch_size=batchSize,epochs=(stop_epoch+1)*2,verbose=verbose,callbacks=callbacks_names)
                    History.append(history2)
                    Model.load_weights(callbacks['checkpoint_valid'].filepath)
                    preds = Model.predict(x_val)
                    if(autoencoder):
                        acc = get_accuracy(preds[1],y_val[1],decimals=2)
                    else:
                        acc = get_accuracy(preds,y_val,decimals=2)
                    return Model, History , acc

                elif validation_mode=='schirrmeister2021':


                    callbacks_names = [callbacks['checkpoint_valid'],
                                        callbacks['early_stopping_valid']]
                    
                    if(autoencoder):
                        history= Model.fit(X_train,Y_train,validation_data=(x_val,y_val),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                    else:
                        history= Model.fit(X_train,Y_train,validation_data=(x_val,y_val),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                    
                    History.append(history)
                    Model.load_weights(callbacks['checkpoint_valid'].filepath)

                    preds = Model.predict(x_val)
                    if(autoencoder):
                        acc = get_accuracy(preds[1],y_val[1],decimals=2)
                    else:
                        acc = get_accuracy(preds,y_val,decimals=2)

                    return Model, History, acc

                    
                elif validation_mode=='lawhern2018':

                    """
                    NO ESTA ADECUADO PARA AUTOENCODERS
                    ----------------------------------
                    """
                   
                    preds = []
                    y_true = []
                    acc = []
                    c = 0

                    skf = StratifiedKFold(n_splits=4)
                    print("data genial: ")

                    if(autoencoder):
                        for train_index, test_index in skf.split(X_train, Y_train[1]):
                            print("data: ",train_index, test_index)
                            X_train_, X_test_ = X_train[train_index], X_train[test_index]
                            y_train_, y_test_ = Y_train[1][train_index], Y_train[1][test_index]

                            x_tr, x_v, y_tr, y_v = train_test_split(X_train_, y_train_, test_size=0.3,random_state=seed)

                            callbacks_names = [callbacks['checkpoint_train'+str(c+1)],callbacks['early_stopping_train'+str(c+1)]]

                            history= Model.fit(x_tr,[x_tr,tf.keras.utils.to_categorical(y_tr,num_classes=2)],validation_data=(x_v, [x_v,tf.keras.utils.to_categorical(y_v,num_classes=2)]),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                            History.append(history)
                            
                            Model.load_weights(callbacks['checkpoint_train'+str(c+1)].filepath)
                            pred = Model.predict(X_test_)
                            preds.append(pred[1])
                            y_preds = preds[c].argmax(axis = -1)
                            y_true.append(y_test_)
                            acc.append(np.mean(y_preds == y_test_))
                            print("Fold %d Classification accuracy: %f " % (c+1,acc[c]))
                            c += 1

                        preds = np.concatenate(preds,axis=0)
                        y_true = np.concatenate(y_true,axis=0)
                        acc = get_accuracy(preds,tf.keras.utils.to_categorical(y_true,num_classes=2),decimals=2)

                        return Model,History,acc
                    else:
                        for train_index, test_index in skf.split(X_train, Y_train):
                            print("data: ",train_index, test_index)
                            X_train_, X_test_ = X_train[train_index], X_train[test_index]
                            y_train_, y_test_ = Y_train[train_index], Y_train[test_index]

                            x_tr, x_v, y_tr, y_v = train_test_split(X_train_, y_train_, test_size=0.3,random_state=seed)

                            callbacks_names = [callbacks['checkpoint_train'+str(c+1)],callbacks['early_stopping_train'+str(c+1)]]

                            history= Model.fit(x_tr,tf.keras.utils.to_categorical(y_tr,num_classes=len(np.unique(Y_train))),validation_data=(x_v, tf.keras.utils.to_categorical(y_v,num_classes=len(np.unique(Y_train)))),batch_size=batchSize,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                            History.append(history)
                            
                            Model.load_weights(callbacks['checkpoint_train'+str(c+1)].filepath)
                            pred = Model.predict(X_test_)
                            preds.append(pred)
                            y_preds = preds[c].argmax(axis = -1)
                            y_true.append(y_test_)
                            acc.append(np.mean(y_preds == y_test_))
                            print("Fold %d Classification accuracy: %f " % (c+1,acc[c]))
                            c += 1

                        preds = np.concatenate(preds,axis=0)
                        y_true = np.concatenate(y_true,axis=0)
                        acc = get_accuracy(preds,tf.keras.utils.to_categorical(y_true,num_classes=len(np.unique(Y_train))),decimals=2)

                        return Model,History,acc



