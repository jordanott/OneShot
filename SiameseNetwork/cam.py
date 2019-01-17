from loader import Conv_Loader
from model import c_net, VGG
from vis.visualization import overlay
from vis.visualization.saliency import visualize_cam

import matplotlib.pyplot as plt
import numpy as np
import json
import os

shape = (250,250,3)
NUM_SAMPLES = 50
TYPE = 'Shallow/'
RESULTS_DIR=TYPE+'CAM_{}/'.format(NUM_SAMPLES)

# create cam dir
if not os.path.exists(RESULTS_DIR): os.mkdir(RESULTS_DIR)

# load data
loader = Conv_Loader(100)
(inputs,targets) = loader.get_batch()
y = np.argmax(targets,axis=1)

# load model
if 'VGG' in RESULTS_DIR: net = VGG(shape,2)
else: net = c_net() #if not LOAD_FROM_AE else load_from_ae('ae.h5',SIAMESE)

#net.load_weights(TYPE+'Results/{}/0/'.format(NUM_SAMPLES))

# predict data
probs = net.predict(inputs)
preds = np.argmax(probs,axis=1)

# cam
for i in range(len(targets)):
    file_path = RESULTS_DIR + '{}_pred_{}_true_{}'.format(i,preds[i],y[i])

    cam = visualize_cam(net,len(net.layers)-1,preds[i],inputs[i].reshape((1,)+shape))
    img = inputs[i]

    plt.imshow(overlay(cam,img*255.))
    plt.savefig(file_path)
