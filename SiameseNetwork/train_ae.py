import json
import numpy as np

from model import ae_net
from keras.preprocessing.image import load_img

DATA_DIR = '../DataGeneration/'
file_path = DATA_DIR+'data.json'

print("loading data from {}".format(file_path))
with open(file_path,"rb") as f:
    data = json.load(f)

def load_location(img_location):
    img = np.array(load_img(img_location))[:,:,0].reshape(200,200,1)
    return img/255.0

x_train = []

ae = ae_net()

for i_key in data:
    m = len(data[i_key])
    samples = np.random.randint(0,m,size=(5000))

    for i in samples:
        img = load_location(DATA_DIR+data[i_key][i])
        x_train.append(img)
        break
    break

x_train = np.array(x_train)

ae.fit(x_train,x_train, epochs=25, batch_size=25)
ae.save_weights('ae.h5')
