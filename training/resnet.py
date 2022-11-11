from keras_preprocessing import image
from keras.models import Input, Model
from keras.layers import Conv2D
import numpy as np
from keras import backend as K
import tensorflow as tf
import keras
import time
import matplotlib.pyplot as plt
import itertools
import json
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

def recall_m(y_true : int, y_pred : int) -> int:
    """Recall metric.

    Args:
        y_true (int): True labels.
        y_pred (int): Predicted labels.

    Returns:
        int: Recall metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true : int, y_pred : int) -> int:
    """Precision metric.

    Args:
        y_true (int): True labels.
        y_pred (int): Predicted labels.

    Returns:
        int: Precision metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true : int, y_pred : int) -> int:
    """F1 metric.

    Args:
        y_true (int): True labels.
        y_pred (int): Predicted labels.

    Returns:
        int: _description_
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class TimeHistory(keras.callbacks.Callback):
    """Keep track of time per epoch."""
    
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
    
    
def get_generator(directory : str, batch_size : int, img_size: tuple, seed : int, validation_split = 0.2) -> tuple:
    """Get a train and validation image generator from a directory.
    
    Args:
        directory (str): Directory containing the images.
        batch_size (int): The batch size that will be used for training.
        img_size (tuple(int,int)): The size to resize the images to.
        seed (int): The seed to use for the random generator.
        validation_split (float, optional): The percentage of images to use for validation. Defaults to 0.2.
        
    Returns:
        tuple: A tuple containing the train and validation image generator.
    """
    train_datagen = ImageDataGenerator(validation_split=validation_split)
    
    train_generator = train_datagen.flow_from_directory(directory,target_size=(img_size,img_size),
                                                        batch_size=batch_size,class_mode='categorical',
                                                        subset='training',seed=seed)
    
    validation_generator = train_datagen.flow_from_directory(directory,target_size=(img_size,img_size),
                                                             batch_size=batch_size,class_mode='categorical',
                                                             subset='validation',seed=seed)
    return train_generator, validation_generator

def get_resnet_model(num_classes : int, img_size : tuple, learning_rate : float) -> Model:
    """Get a pretrained Resnet 50 model. 

    Args:
        num_classes (int): The number of classes to train on.
        img_size (tuple): The size of the images.
        learning_rate (float): The learning rate to use for training.

    Returns:
        Model: The pretrained Resnet 50 model.
    """
    # Grab pretrained model, include_top removes the classification layer
    ResNet50_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, classes=num_classes, input_shape=(img_size,img_size,3))
    
    # Layers are frozen by default, performance seems to tank if we freeze them
    for layer in ResNet50_model.layers:
        layer.trainable = True
    
    # Creating fully connected layer for learning
    resnet50_x = tf.keras.layers.Flatten()(ResNet50_model.output)
    resnet50_x = tf.keras.layers.Dense(512,activation='relu')(resnet50_x)
    resnet50_x = tf.keras.layers.Dense(num_classes,activation='softmax')(resnet50_x)
    resnet50_x_final_model = tf.keras.Model(inputs=ResNet50_model.input, outputs=resnet50_x)
    
    opt = tf.keras.optimizers.SGD(lr=0.01,momentum=0.7)
    return resnet50_x_final_model, opt

def get_callbacks(file_path : str, fold : int) -> list:
    """Get a list of callbacks to use for training.

    Args:
        file_path (str): The path to save the model to.

    Returns:
        list: A list of callbacks to use for training.
    """
    time_callback = TimeHistory()
    resnet_filepath = file_path+'fold'+str(fold)+'-resnet50v2-saved-model-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5'
    resnet_checkpoint = tf.keras.callbacks.ModelCheckpoint(resnet_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    resnet_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=0.000002)
    tb_callback = tf.keras.callbacks.TensorBoard(file_path+'tb_logs', update_freq=1)
    return [resnet_checkpoint,resnet_early_stopping,reduce_lr,tb_callback,time_callback]


def train_resnet_model_k_fold(num_classes : int, img_size : tuple,train_data_location : str, number_of_epochs : int,
                              file_path : str, num_folds : int = 5, batch_size : int = 64) -> None:
    """Train a Resnet 50 model using k-fold cross validation.

    Args:
        num_classes (int): The number of classes to train on.
        img_size (tuple): The size of the images.
        train_data_location (str): The location of the training data.
        number_of_epochs (int): The number of epochs to train for.
        file_path (str): The path to save the model to.
        num_folds (int, optional): The number of folds to use for cross validation. Defaults to 5.
        batch_size (int, optional): The batch size to use for training. Defaults to 64.
    """
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index='0')):
        for fold in range(num_folds):
            print("Fold: ", fold)
            train_generator, validation_generator = get_generator(train_data_location, batch_size, img_size,fold)
            resnet50_x_final_model, opt = get_resnet_model(num_classes, img_size, 0.01)
            callback_list = get_callbacks(file_path, fold)
            
            resnet50_x_final_model.compile(loss = 'categorical_crossentropy', optimizer= opt, 
                                metrics=['acc',f1_m,precision_m, recall_m,tf.keras.metrics.AUC(),tf.keras.metrics.FalseNegatives(),
                                    tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(),
                                    tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
            
            resnet50_history = resnet50_x_final_model.fit(train_generator, epochs = number_of_epochs ,validation_data = validation_generator,callbacks=callback_list,verbose=1)
            pd.DataFrame.from_dict(resnet50_history.history).to_csv(file_path+'history'+str(fold)+'.csv',index=False)



def train_resnet_model(num_classes, img_size,train_generator,number_of_epochs, validation_generator, file_path):
    time_callback = TimeHistory()
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index='0')):
        # Grab pretrained model, include_top removes the classification layer
        ResNet50_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, classes=num_classes, input_shape=(img_size,img_size,3))
        
        # Resnet wants a three chanel input, but we have grayscale images
        #input_tensor = Input(shape=(img_size,img_size,1))
        # x has a dimension of (IMG_SIZE,IMG_SIZE,3)
        #x = Conv2D(3,target_size,padding='same')(input_tensor) 
        # Prepend the Resnet model with the input tensor
        #out = original_ResNet50_model(x) 
        #ResNet50_model = Model(inputs=input_tensor,outputs=out)
        
        # Layers are frozen by default, performance seems to tank if we freeze them
        for layer in ResNet50_model.layers:
            layer.trainable = True
        
        # Creating fully connected layer for learning
        resnet50_x = tf.keras.layers.Flatten()(ResNet50_model.output)
        resnet50_x = tf.keras.layers.Dense(512,activation='relu')(resnet50_x)
        resnet50_x = tf.keras.layers.Dense(num_classes,activation='softmax')(resnet50_x)
        resnet50_x_final_model = tf.keras.Model(inputs=ResNet50_model.input, outputs=resnet50_x)
        
        #opt = tf.keras.optimizers.Adam(lr=0.01)
        opt = tf.keras.optimizers.SGD(lr=0.01,momentum=0.7)
        resnet50_x_final_model.compile(loss = 'categorical_crossentropy', optimizer= opt, 
                                    metrics=['acc',f1_m,precision_m, recall_m,tf.keras.metrics.AUC(),tf.keras.metrics.FalseNegatives(),
                                        tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(),
                                        tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        resnet_filepath = file_path+'resnet50v2-saved-model-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5'
        resnet_checkpoint = tf.keras.callbacks.ModelCheckpoint(resnet_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        resnet_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=0.000002)
        tb_callback = tf.keras.callbacks.TensorBoard(file_path+'tb_logs', update_freq=1)
        callbacklist = [resnet_checkpoint,resnet_early_stopping,reduce_lr,tb_callback,time_callback]

        resnet50_history = resnet50_x_final_model.fit(train_generator, epochs = number_of_epochs ,validation_data = validation_generator,callbacks=callbacklist,verbose=1)
        times = time_callback.times
        
        #history_dict = resnet50_history.history
        #json.dump(history_dict, open(file_path+'history_dump.json', 'w'))
        pd.DataFrame.from_dict(resnet50_history.history).to_csv(file_path+'history.csv',index=False)
        
        return resnet50_x_final_model, resnet50_history,times

