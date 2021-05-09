""" This is a file to keep the directories for the training and validation data

Variables (to be imported) :
    x_path: Path of the train X data
    y_path: Path of the train Y data
    x_test_path: Path of the validation X data
    y_test_path: Path of the validation Y data
"""

from itertools import chain
from os import listdir
from cityscapesscripts.helpers.csHelpers import getCoreImageFileName
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import one_hot
from random import randint

from time import perf_counter

def get_dirs(path):
    paths = listdir(path)
    return [path + "/" + p for p in paths]

x = "leftImg8bit/train"
x = get_dirs(x)
train_names = list(map(getCoreImageFileName, chain(*list(map(listdir,x)))))
y = "leftImg8bit/val"
y = get_dirs(y)
val_names = list(map(getCoreImageFileName, chain(*list(map(listdir,y)))))


"""The directories for the basic (not extra) images
"""
x_path = [f"leftImg8bit/train/{n.split('_')[0]}/{n}_leftImg8bit.png" for n in train_names]
y_path = [f"gtCoarse/train/{n.split('_')[0]}/{n}_gtCoarse_labelIds.png" for n in train_names]

x_test_path = [f"leftImg8bit/val/{n.split('_')[0]}/{n}_leftImg8bit.png" for n in val_names]
y_test_path = [f"gtCoarse/val/{n.split('_')[0]}/{n}_gtCoarse_labelIds.png" for n in val_names]

def get_train_data(train=None, test=None):
    """Obtain a certain number of training and validation data from the dataset.

    Args:
        train (int, optional): The number of training data requested. Defaults to None.
        test (int, optional): The number of validation data requested. Defaults to None.

    Returns:
        (np.array, np.array), (np.array, np.array): The training and testing data processed as numpy array of the images. 
        The pixels will be normalized between 0 and 1.
    """
    x_train = np.array(list(map(lambda d : np.asarray(Image.open(d))/255, x_path[:train])))
    y_train = np.array(list(map(lambda d : np.asarray(Image.open(d)), y_path[:train])))

    x_test = np.array(list(map(lambda d : np.asarray(Image.open(d))/255, x_test_path[:test])))
    y_test = np.array(list(map(lambda d : np.asarray(Image.open(d)), y_test_path[:test])))

    return (x_train, y_train), (x_test, y_test)

def relabel(im, downscale):
    """Downscales a label for an image using the most frequently occuring block in the image.

    Args:
        im (np.array): A feature map. Must be 1024x2048x1 pixels
        downscale (int): The downscale factor. Must be 2,4,8 or 16x

    Raises:
        ValueError: If the downscale factor is not supported

    Returns:
        np.array: The downscaled feature map.
    """
    if downscale != 2 and downscale != 4 and downscale != 8 and downscale != 16:
        raise ValueError("Can only downscale by 2, 4, 8, or 16x")
    
    res = np.empty((1024//downscale,2048//downscale), dtype=np.uint8)
    for i in range(1024//downscale):
        for j in range(2048//downscale):
            res[i,j] = np.bincount(im[i*downscale:(i+1)*downscale, j*downscale:(j+1)*downscale].flatten()).argmax()
    return res

def generate_train(downscale=4, batch_size=2):
    for i in range(len(x_path)):
        if downscale==1:
            x_train = np.asarray(Image.open(x_path[i]))/255
            y_train = np.asarray(Image.open(y_path[i]))
        elif randint(0, 1):
            x_train = np.flip(np.asarray(Image.open(x_path[i]).resize((2048//downscale, 1024//downscale))),axis=1)/255
            y_train = np.flip(np.asarray(Image.open(y_path[i]).resize((2048//downscale, 1024//downscale), resample=Image.NEAREST)), axis=1)
        else:
            x_train = np.asarray(Image.open(x_path[i]).resize((2048//downscale, 1024//downscale)))/255
            y_train = np.asarray(Image.open(y_path[i]).resize((2048//downscale, 1024//downscale), resample=Image.NEAREST))
            
        x_train = x_train.reshape(1, 1024//downscale, 2048//downscale, 3)
        y_train = tf.reshape(one_hot(y_train, 30), (1, 1024//downscale, 2048//downscale, 30))

        if i == 0:
            x = x_train
            y = y_train
        elif i % batch_size == 0:
            yield x, y
            x = x_train
            y = y_train
        else:
            x = np.append(x, x_train, 0)
            y = np.append(y, y_train, 0)
        
            


def generate_test(downscale=4, batch_size=2):
    for i in range(len(x_test_path)):
        if downscale==1:
            x_train = np.asarray(Image.open(x_test_path[i]))/255
            y_train = np.asarray(Image.open(y_test_path[i]))
        else:
            x_train = np.flip(np.asarray(Image.open(x_test_path[i]).resize((2048//downscale, 1024//downscale))),axis=1)/255
            y_train = np.flip(np.asarray(Image.open(y_test_path[i]).resize((2048//downscale, 1024//downscale), resample=Image.NEAREST)), axis=1)
            
        x_train = x_train.reshape(1, 1024//downscale, 2048//downscale, 3)
        y_train = tf.reshape(one_hot(y_train, 30), (1, 1024//downscale, 2048//downscale, 30))

        if i == 0:
            x = x_train
            y = y_train
        elif i % batch_size == 0:
            yield x, y
            x = x_train
            y = y_train
        else:
            x = np.append(x, x_train, 0)
            y = np.append(y, y_train, 0)

if __name__ == '__main__':
    print(len(x_path))
    print(len(y_path))
    print(len(x_test_path))
    print(len(y_test_path))

    tr = generate_train()
    te = generate_test()

    x_train, y_train = next(tr)
    print(f"X_TRAIN has shape {x_train.shape}, X_TRAIN looks like {x_train}.")
    print(f"Y_TRAIN has shape {y_train.shape}, Y_TRAIN looks like {y_train}.\nThe maximum of Y_TRAIN is {np.max(y_train)}")
    print(next(te))
