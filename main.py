"""
This module defines tbe training of the model based on the various hyperparameters input by the user.
"""
import os
import csv
import argparse

from keras import callbacks
from keras.utils import plot_model
from keras.utils import to_categorical

from sklearn.preprocessing import normalize

from data import *
from Convnet import * 

# setting the hyper parameters and other configurations
parser = argparse.ArgumentParser(description="1D Convolutional Network on Histone Modification Signals.")
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.9, type=float,
                help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
parser.add_argument('--data_dir', default='./data',
                help="The directory of the data files")
parser.add_argument('--debug', action='store_true',
                help="Save weights by TensorBoard")
parser.add_argument('--save_dir', default='./without_spat')
parser.add_argument('--mix', default=False,
                help="Train on the data by mixing all cell types")
# parser.add_argument('-w', '--weights', default=None,
                # help="The path of the saved weights. Should be specified when testing")

args = parser.parse_args()
print(args)

# If the path of save_dir does not exist create one
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# If the data is not to be mixed
if not args.mix:

    scores = {}
    cells = os.listdir(args.data_dir)
    count = 1

    # Iterate over all cells individually
    for cell in cells :

        print('-'*50 + 'Start'+ '-'*50)

        print("Cell Number: ",count)
        count += 1
        print("Cell Type: "+cell)

        # Load the individual cell data
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(args.data_dir+'/'+cell+'/classification')

        filepath = args.save_dir+'/'+cell

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # define model
        model = Conv1DNet(input_shape=x_train.shape[1:])

        # Train the model
        model.summary()
        plot_model(model, filepath+'/model.png', show_shapes=True)
        model = train(model=model, data=((x_train, y_train), (x_val, y_val)), args=args, dirs=filepath)

        # Test the model
        test_model = models.load_model(filepath+'/model.h5')
        auc = test(model=test_model, data=(x_test, y_test), filepath=filepath)
        scores[cell] = auc

        print('-'*50 + ' End '+ '-'*50)
        print(' ')
        print(' ')
        print(' ')

    # write the results to a file
    w = csv.writer(open("output.csv", "w"))
    for key, val in scores.items():
        w.writerow([key, val])

# If training on the complete mixed data
else:

    print('-'*50+' Loading Data '+'-'*50)

    # Load the complete mixed data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data_mix(args)

    print('-'*50+' Finished loading Data '+'-'*50)

    # define the model
    model = Conv1DNet(input_shape=x_train.shape[1:])
    model.summary()
    plot_model(model, args.save_dir+'/model.png', show_shapes=True)

    # Train the model
    model = train(model=model, data=((x_train, y_train), (x_val, y_val)), args=args, dirs=args.save_dir)

    # test_model = models.load_model(args.save_dir+'/model.h5')

    print('-'*50+' Testing Begins '+'-'*50)

    scores = {}
    cells = os.listdir(args.data_dir)
    count = 1
    for cell in cells :

        print('-'*50 + 'Start'+ '-'*50)
        print("Cell Number: ",count)
        count += 1
        print("Cell Type: "+cell)

        filepath = args.data_dir+'/'+cell+'/classification'
        
        # Load data and transform it such that it can be fed to the model
        test_data = pd.read_csv(filepath+'/test.csv', header=None)

        y_test = test_data[7]
        x_test = test_data.drop([0, 1, 7], axis=1)

        x_test = np.array(x_test, dtype=float)
        y_test = np.array(y_test, dtype=float)

        y_test = y_test.reshape((y_test.shape[0]//100, 100))
        y_test = y_test[:, 0]
        x_test = x_test.reshape((x_test.shape[0]//100,100,x_test.shape[1]))

        for i in range(x_test.shape[0]):
            s = x_test[i,:,:]
            s = normalize(s, axis=1, norm='max')
            x_test[i,:,:] = s

        a = x_test.sum(1)
        a = normalize(a, axis=1, norm='max')
        x_test = x_test.reshape((x_test.shape[0],-1))
        x_test = np.c_[x_test, a]
        x_test = x_test.reshape((x_test.shape[0],-1,1))

        y_test = to_categorical(y_test)


        if not os.path.exists(args.save_dir+'/'+cell):
            os.makedirs(args.save_dir+'/'+cell)

        # Test the model
        auc = test(model=model, data=(x_test, y_test), filepath=args.save_dir+'/'+cell)
        scores[cell] = auc

        print('-'*50 + ' End '+ '-'*50)
        print(' ')
        print(' ')
        print(' ')

    # write the results to a file
    w = csv.writer(open(args.save_dir+'/output-mix.csv', "w"))
    for key, val in scores.items():
        w.writerow([key, val])
