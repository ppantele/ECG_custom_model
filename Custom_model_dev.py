"""
"Custom model" for ECG interpretation
Developed in April, 2021,
for the SBDSO MSc program, 
DSV, University of Stockholm.

@author: Panteleimon Pantelidis
@email: pan.g.pantelidis@gmail.com
"""

import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.metrics import Recall, Precision, AUC




class Custom_model:

  def __init__(self, 
               input_shape, 
               n_classes,
               verbose = True, 
               batch_size = 64, 
               n_filters = [16, 32, 128],  # alternatively: [8, 16, 64] or [32, 64, 256]
               kernel_size = [2, 20, 80],  # alternatively: [2, 10, 40] or [2, 40, 150]      
               n_epochs = 1500,
               early_stop = 7,
               pred_threshold = 0.5):
    '''
    input_shape: tuple (data_points x number of time-series, eg. 500x12 for ECG)
    n_classes: int
    '''
    
    self.n_filters = n_filters
    self.kernel_size = kernel_size
    self.callbacks = None
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.early_stop = early_stop
    self.verbose = verbose
    self.pred_threshold = pred_threshold

    self.model = self._create_model(input_shape, n_classes)
    if (verbose == True):
      self.model.summary()
      
  

  def _proc_arm(self, input, depth):

    kernels = self.kernel_size
    filters = self.n_filters[depth]
    x = input

    for i in range(2):
      conv_list = []
      for kernel in kernels:
        conv_list.append(keras.layers.Conv1D(filters=filters,
                                             kernel_size=kernel,
                                             strides=1, 
                                             padding='same', 
                                             activation='relu', 
                                             use_bias=False)(x))
      x = keras.layers.Concatenate(axis=2)(conv_list)
      x = keras.layers.BatchNormalization()(x)
      x = keras.layers.Activation(activation='relu')(x)

    return x


  
  def _res_arm(self, input, output):
        
    shortcut_x = keras.layers.Conv1D(filters=int(output.shape[-1]),
                                     kernel_size=1,
                                     padding='same',
                                     use_bias=False)(input)                                     
    shortcut_x = keras.layers.normalization.BatchNormalization()(shortcut_x)

    x = keras.layers.Add()([shortcut_x, output])
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)

    return x



  def _create_model(self, input_shape, n_classes):

    input_layer = keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for block_index in range(3):                  # 3 Res blocks

      x = self._proc_arm(x, depth=block_index)
      x = self._res_arm(input_res, x)
      input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)
    output_layer = keras.layers.Dense(n_classes, activation='sigmoid')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', Recall(), Precision(), AUC()])
    
    red_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    
    if type(self.early_stop)==int:
        es = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                           patience=self.early_stop, 
                                           restore_best_weights=True)
        self.callbacks = [red_lr, es]
    else:
        self.callbacks = [red_lr]
    
    return model



  def fit(self, x_train, y_train):

    model_trained = self.model.fit(x_train, 
                                   y_train, 
                                   batch_size=self.batch_size, 
                                   epochs=self.n_epochs,
                                   verbose=self.verbose, 
                                   validation_split = 0.05,
                                   callbacks=self.callbacks)
    return model_trained



  def predict(self, x_test):
      y_pred = self.model.predict(x_test, batch_size=self.batch_size)
      y_pred = (y_pred>=self.pred_threshold).astype(int)
      return y_pred
