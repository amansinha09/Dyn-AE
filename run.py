#load dataset
import os, sys
'''
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
'''



#include required libraries
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import os
import numpy as np
from time import time
from DynAE import DynAE
from datasets import load_data, load_data_conv
import metrics


#Parameter setting
n_clusters = 2
dataset = 'flicker'
loss_weight_lambda = 0.5
save_dir='home/aman/Dyn-AE/results'
visualisation_dir='home/aman/Dyn-AE/visualisation'
data_path = 'home/aman/Dyn-AE/flickr-sarcasm-dataset/train/transformed'
#'/content/drive/My Drive/Colab Notebook/Dyn-AE/data/' + dataset
batch_size = 16
maxiter_pretraining = 5000
maxiter_clustering = 5000
tol=0.01
optimizer1=SGD(0.001, 0.9)#0.001
optimizer2=tf.train.AdamOptimizer(0.0001)#0.0001
kappa = 3
ws=0.1
hs=0.1
rot=10
scale=0.




#load and initialize the model
x, y = load_data(dataset, data_path)
model = DynAE(batch_size=batch_size, dataset=dataset, dims=[x.shape[-1], 500, 500, 2000, 10], loss_weight=loss_weight_lambda, n_clusters=n_clusters, visualisation_dir=visualisation_dir, ws=ws, hs=hs, rot=rot, scale=scale)
model.compile_dynAE(optimizer=optimizer1)
model.compile_disc(optimizer=optimizer2)
model.compile_aci_ae(optimizer=optimizer2)



#uncomment and Load the pretraining weights if you have already pretrained your network

#model.ae.load_weights(save_dir + '/' + dataset + '/ae_weights.h5')
#model.critic.load_weights(save_dir + '/' + dataset + '/critic_weights.h5')




#Pre-training

model.train_aci_ae(x, y, maxiter=maxiter_pretraining, batch_size=batch_size, validate_interval=500, save_interval=100, save_dir=save_dir, verbose=1, aug_train=True)


#clustering phase
y_pred = model.train_dynAE(x=x, y=y, kappa=kappa, n_clusters=n_clusters, maxiter=maxiter_clustering, batch_size = 32, tol=tol, validate_interval=50, show_interval=None, save_interval=50, save_dir=save_dir, aug_train=True)



