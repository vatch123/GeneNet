"""
This module defines the model architecture along with the train and test loops.
"""

import numpy as np
import pandas as pd

from keras import layers, models, optimizers
from keras import backend as K

import keras.callbacks as callbacks

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc


from utils import plot_log


def Conv1DNet(input_shape):
    """
    This function defines the model architecture and returns it.

    Parameters
    ----------
    input_shape : `tuple`
        The shape of the input data
    """
   

    x = layers.Input(shape=input_shape)
    conv0 = layers.Conv1D(filters=64, kernel_size=17, strides=1, padding='valid', activation='tanh', name='conv0')(x)
    pool0 = layers.MaxPooling1D(pool_size=3)(conv0)

    # The spatial dropout layer was removed for the experimentation
    # drop0 = layers.SpatialDropout1D(0.4)(pool0)

    conv2 = layers.Conv1D(filters=128, kernel_size=13, strides=1, padding='valid', activation='tanh', name='conv2')(pool0)
    conv3 = layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='valid', activation='tanh', name='conv3')(conv2)
    pool1 = layers.MaxPooling1D(pool_size=4)(conv3)
    drop1 = layers.Dropout(0.6)(pool1)

    
    # conv4 = layers.Conv1D(filters=512, kernel_size=7, strides=2, padding='valid', activation='relu', name='conv4')(drop1)

    flat = layers.Flatten()(drop1)
    dense1 = layers.Dense(35, activation='tanh')(flat)
    drop2 = layers.Dropout(0.5)(dense1)
    
    dense2 = layers.Dense(16, activation='tanh')(drop2)
    dense3 = layers.Dense(8, activation='tanh')(dense2)
    out = layers.Dense(2, activation='softmax')(dense3)

    train_model = models.Model(input=x, output=out)

    return train_model


def train(model, data, args, dirs):
    """
    The function which defines the training loop of the model

    Parameters
    ----------
    model : `keras.models.Model`
        The structure of the model which is to be trained
    data : `tuple`
        The training and validation data
    args : `dict`
        The argument dictionary which defines other parameters at training time
    dirs : `string`
        Filepath to store the logs
    """

    # Extract the data
    (x_train, y_train), (x_val, y_val) = data

    # callbacks
    log = callbacks.CSVLogger(dirs + '/log.csv')

    tb = callbacks.TensorBoard(log_dir=dirs + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    
    checkpoint = callbacks.ModelCheckpoint(dirs + '/model.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=False, verbose=1)
    
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    
    # Training without data augmentation:
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
              validation_data=(x_val, y_val), callbacks=[log, tb, checkpoint, lr_decay]) #, roc_auc_callback((x_train, y_train), (x_val, y_val))])

    # Save the trained model
    model.save(dirs + '/trained_model.h5')

    # Plot the training results
    plot_log(dirs, show=False)

    return model


def test(model, data, filepath):
    """
    Function to test the performance of the trained model on the test data

    Parameters
    ----------
    model : `keras.models.Model`
        The trained model
    data : `tuple`
        The test data
    filepath : `string`
        The filepath to store the results of the tests
    
    Returns
    -------
    `float`
        The AUC score
    """

    # Extract the data
    x_test, y_test = data

    y_pred = model.predict(x_test, batch_size=100, verbose=1)
    
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    
    fpr, tpr, thresholds = roc_curve(y_test[:,1], y_pred[:,1])
    auc_score = roc_auc_score(y_test[:,1], y_pred[:,1])
    
    print('auc Score: ', auc_score)


    # plot the roc curve
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Area under Curve (area = {:.3f})'.format(auc_score))
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    
    plt.savefig(filepath+"/roc.png")
    plt.close()

    return auc_score
