import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split,StratifiedKFold

class train_model_cv():
    def __init__(self,model):
        super(train_model_cv,self).__init__()
        self.model = model


    def fit_model(self,X,y,X_val,y_val,batch_size,epochs,verbose,callbacks,retrain=False):
        if retrain==False:
            self.create_model()
        history= self.model.fit(X,y,validation_data=(X_val,y_val),batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks)
        return history

    def predict(self,X):
        preds = self.model.predict(X)
        return preds

    def fit_validation(self,X,y,X_val=None,y_val=None,batch_size=64,epochs=1000,verbose=1,val_mode=None,autoencoder=False,early_stopping=False,triplet_loss=False,model_name=None):
        History = []
        num_classes = len(np.unique(y))
        if val_mode=='schirrmeister2017':

            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            y_tr= tf.keras.utils.to_categorical(y_tr,num_classes=num_classes)
            y_ts= tf.keras.utils.to_categorical(y_ts,num_classes=num_classes)

            callbacks_names = [self.callbacks['early_stopping_train'],self.callbacks['checkpoint_train']]

            if autoencoder and not triplet_loss:
                y_tr = [X_tr,y_tr]
                y_ts = [X_ts,y_ts]
            elif autoencoder and triplet_loss:
                print('aqui')
                y_tr = [X_tr,np.argmax(y_tr,axis=1),y_tr]
                y_ts = [X_ts,np.argmax(y_ts,axis=1),y_ts]
                #callbacks_names.append(self.callbacks['CSVLogger'])
                callbacks_names.append(self.callbacks['reduce_lr_train'])

            history1 = self.fit_model(X_tr, y_tr,X_ts, y_ts,batch_size=batch_size,epochs=epochs,
                                        verbose=verbose,callbacks=callbacks_names)
            History.append(history1)
            stop_epoch= np.argmin(history1.history['val_loss'])
            loss_stop = history1.history['loss'][stop_epoch]


            self.model.load_weights(self.callbacks['checkpoint_train'].filepath)

            self.callbacks['Threshold_valid'].threshold = loss_stop
            self.callbacks['early_stopping_valid'].patience = (stop_epoch)*2
            callbacks_names = [self.callbacks['Threshold_valid'],self.callbacks['checkpoint_valid'],
                               self.callbacks['early_stopping_valid']]

            y_train= tf.keras.utils.to_categorical(y,num_classes=num_classes)
            y_valid= tf.keras.utils.to_categorical(y_val,num_classes=num_classes)

            if autoencoder and not triplet_loss:
                y_train = [X,y_train]
                y_valid = [X_val,y_valid]
            elif autoencoder and triplet_loss:
                y_train = [X,np.argmax(y_train,axis=1),y_train]
                y_valid = [X_val,np.argmax(y_valid,axis=1),y_valid]
                #callbacks_names.append(self.callbacks['CSVLogger'])
                callbacks_names.append(self.callbacks['reduce_lr_train'])

            history2= self.fit_model(X,y_train,X_val, y_valid,batch_size=batch_size,epochs=(stop_epoch+1)*2,
                                        verbose=verbose,callbacks=callbacks_names,retrain=True)
            History.append(history2)
            self.model.load_weights(self.callbacks['checkpoint_valid'].filepath)

            if autoencoder:
                self.preds = self.predict(X_val)[-1]
            else:
                self.preds = self.predict(X_val)

            self.y_true = y_val

        elif val_mode=='schirrmeister2017_legal':

            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            y_tr= tf.keras.utils.to_categorical(y_tr,num_classes=num_classes)
            y_ts= tf.keras.utils.to_categorical(y_ts,num_classes=num_classes)

            callbacks_names = [self.callbacks['early_stopping_train']]

            if autoencoder:
                y_tr = [X_tr,y_tr]
                y_ts = [X_ts,y_ts]

            if model_name == 'EEGNet_fusion':
                X_tr = [X_tr,X_tr,X_tr]
                X_ts = [X_ts,X_ts,X_ts]

            history1 = self.fit_model(X_tr, y_tr,X_ts, y_ts,batch_size=batch_size,epochs=epochs,
                                        verbose=verbose,callbacks=callbacks_names)

            History.append(history1)
            stop_epoch= np.argmin(history1.history['val_loss'])
            loss_stop = history1.history['loss'][stop_epoch]

            #self.model.load_weights(self.callbacks['checkpoint_train'].filepath)

            #self.callbacks['Threshold_valid'].threshold = loss_stop
            self.callbacks['early_stopping_valid'].patience = (stop_epoch)*2
            callbacks_names = [self.callbacks['early_stopping_valid']]

            y_train= tf.keras.utils.to_categorical(y,num_classes=num_classes)

            if autoencoder:
                y_train = [X,y_train]

            if model_name == 'EEGNet_fusion':
                X = [X,X,X]

            history2= self.fit_model(X,y_train,X_ts, y_ts,batch_size=batch_size,epochs=(stop_epoch+1)*2,
                                                            verbose=verbose,callbacks=callbacks_names,retrain=True)
            History.append(history2)
            #self.model.load_weights(self.callbacks['checkpoint_valid'].filepath)

            if autoencoder:
                self.preds = self.predict(X_val)[-1]
            elif model_name == 'EEGNet_fusion':
                self.preds = self.predict([X_val,X_val,X_val])
            else:
                self.preds = self.predict(X_val)

            self.y_true = y_val
            return History

        elif val_mode=='schirrmeister2021':

            y_train= tf.keras.utils.to_categorical(y,num_classes=num_classes)
            y_valid= tf.keras.utils.to_categorical(y_val,num_classes=num_classes)

            callbacks_names = [self.callbacks['checkpoint_valid'],
                               self.callbacks['early_stopping_valid']]

            if autoencoder:
                y_train = [X,y_train]
                y_valid = [X_val,y_valid]

            history= self.fit_model(X,y_train,X_val, y_valid,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
            History.append(history)

            self.model.load_weights(self.callbacks['checkpoint_valid'].filepath)

            if autoencoder:
                self.preds = self.predict(X_val)[-1]
            else:
                self.preds = self.predict(X_val)

            self.y_true = y_val
        elif val_mode=='lawhern2018':

            preds = []
            y_true = []
            acc = []
            c = 0

            skf = StratifiedKFold(n_splits=4)

            for train_index, test_index in skf.split(X, y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_tr, X_valid, y_tr, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=self.seed)

                # if early_stopping:
                #     callbacks_names = [self.callbacks['early_stopping_train'+str(c+1)]]
                # else:
                #     ##callbacks_names = [self.callbacks['checkpoint_train'+str(c+1)]]
                #     pass

                y_valid = tf.keras.utils.to_categorical(y_valid,num_classes=num_classes)
                y_tr = tf.keras.utils.to_categorical(y_tr,num_classes=num_classes)

                if autoencoder:
                    y_tr    = [X_tr,y_tr]
                    y_valid = [X_valid,y_valid]

                #history= model.fit(X_tr,y_tr,validation_data=(X_val,y_val),batch_size=16,epochs=500,verbose=0,callbacks=[checkpointer],class_weight=class_weights)
                history= self.fit_model(X_tr,y_tr,X_valid, y_valid,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=None)
                History.append(history)

                ##self.model.load_weights(self.callbacks['checkpoint_train'+str(c+1)].filepath)

                if autoencoder:
                    pred = self.predict(X_test)[-1]
                else:
                    pred = self.predict(X_test)

                preds.append(pred)
                y_preds = preds[c].argmax(axis = -1)
                y_true.append(y_test)
                acc.append(np.mean(y_preds == y_test))
                print("Fold %d Classification accuracy: %f " % (c+1,acc[c]))
                c += 1

            self.preds = np.concatenate(preds,axis=0)
            self.y_true = np.concatenate(y_true,axis=0)

            return History

    def get_pred_labels(self):
        pred_labels = np.argmax(self.preds,axis=-1)
        return pred_labels

    def get_accuracy(self,decimals=2):
        pred_labels = self.get_pred_labels()
        acc = np.mean(pred_labels==self.y_true)
        return np.round(acc*100,decimals=decimals)