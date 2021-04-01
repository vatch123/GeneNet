"""
This module enables plotting of the logged data during the training time.
This has been adapted from the XifengGuo's implementation of Capsule Net. 
"""

import numpy as np
import csv
import math
import pandas
from matplotlib import pyplot as plt


__all__ = ["plot_log"]


def plot_log(filepath, show=False):
    """
    Plot the logged data - the training accuracy, validation accuracy and training loss

    Parameters
    ----------
    filepath : `string`
        The filepath where the log files are stored
    show : `bool`
        Whether to show the plots or not

    """

    # load the logs data
    data = pandas.read_csv(filepath+'/log.csv')

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')


    plt.savefig(filepath+'/log.png')
    if show:
        plt.show()
    plt.close()


if __name__=="__main__":
    plot_log('result/log.csv')
