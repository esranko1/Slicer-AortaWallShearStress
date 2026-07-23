import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import spearmanr
from os.path import isfile, join
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn import metrics, preprocessing
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.colors import LightSource
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as sm
import imageio
import pickle
import mat73
tf.random.set_seed(1)

#data extraction

#define model

#loss and training

#evaluation