from loader import Siamese_Loader
from model import siamese_net
import matplotlib.pyplot as plt
import numpy as np

batch_size = 16

def plot(img1,img2,lang,pred,target,i):
    d = {1:'same',2:'different'}
    f, ax = plt.subplots(1,2)

    ax[0].set_title(lang[0])
    ax[1].set_title(lang[1])
    ax[0].set_xlabel('Predicted {}: {}'.format(d[np.rint(pred)],pred))
    ax[1].set_xlabel('Actual: '+str(target))

    ax[0].set_axis_off()
    ax[1].set_axis_off()

    ax[0].imshow(img1)
    ax[1].imshow(img2)

    plt.savefig('{}.png'.format(i))

loader = Siamese_Loader()
siamese_net.load_weights('weights.h5')

inputs,targets,language_types = loader.get_batch(batch_size,s='test')

predictions = siamese_net.predict(inputs)

print 'Accuracy:',np.sum(np.rint(probs).flatten() == targets)/float(batch_size)

for i in range(0,batch_size):
    pred = predictions[i][0]
    img1,img2 = inputs[i]
    lang = language_types[i]
    target = targets[i]

    plot(img1,img2,lang,pred,target,i)
