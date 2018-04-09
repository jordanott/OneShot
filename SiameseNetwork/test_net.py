from loader import Siamese_Loader
from model import siamese_net
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

n_itr = 5
batch_size = 16

def plot(img1,img2,lang,pred,target,i):
    d = {1:'same',0:'different'}
    f, ax = plt.subplots(1,2)

    ax[0].set_title(lang[0].split('/')[3])
    ax[1].set_title(lang[1].split('/')[3])
    ax[0].set_xlabel('Predicted {}: {}'.format(d[int(np.rint(pred))],pred))
    ax[1].set_xlabel('Actual: '+str(target))

    #ax[0].set_axis_off()
    #ax[1].set_axis_off()

    ax[0].imshow(img1.reshape(200,200))
    ax[1].imshow(img2.reshape(200,200))

    plt.savefig('Images/{}.png'.format(i))

loader = Siamese_Loader()
siamese_net.load_weights('weights.h5')

counter = 0
for itr in range(n_itr):
    inputs,targets,language_types = loader.get_batch(batch_size,s='test')

    predictions = siamese_net.predict(inputs)

    print 'Accuracy:',np.sum(np.rint(predictions).flatten() == targets)/float(batch_size)

    for i in range(0,batch_size):
        pred = predictions[i][0]
        img1,img2 = inputs[0][i],inputs[1][i]
        lang = language_types[i]
        target = targets[i]

        plot(img1,img2,lang,pred,target,counter)
        counter += 1
