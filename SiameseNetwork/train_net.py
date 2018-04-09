from loader import Siamese_Loader
from model import siamese_net
import matplotlib.pyplot as plt
import numpy as np
import json
import os

PATIENCE = 15
batch_size = 16
n_iter = 1000000
best = -1

with open('latex.txt','w') as tex:

    line = 'Language Samples & Train Same & Train Diff & Test Same & Test Diff & Val Acc \\\\\n'
    text.write(line)

for lang_samples in range(1,72,10):
    PATH = str(lang_samples)+'/'
    os.mkdir(PATH)
    evaluate_every = lang_samples*10
    weights_path = PATH + 'weights.h5'

    loader = Siamese_Loader(lang_samples)
    monitor = {
        'train_acc' = [],
        'val_acc' = []
    }
    tmp_train_acc = []

    for i in range(1, n_iter):
        (inputs,targets) = loader.get_batch(batch_size)
        history = siamese_net.fit(inputs,targets)
        tmp_train_acc.append(history.history['acc'][-1])

        if i % evaluate_every == 0:
            val_acc = loader.test_oneshot(siamese_net,batch_size,verbose=True)
            print 'Batches:',i,'Validation accuracy:',val_acc
            monitor['val_acc'].append(val_acc)
            if val_acc >= best:
                print("saving")
                siamese_net.save(weights_path)
                best=val_acc

            train_acc = np.mean(tmp_train_acc)
            tmp_train_acc = []
            monitor['train_acc'].append(train_acc)
            print("Iteration {}, training acc: {:.2f},".format(i,train_acc))
            # if there has been at least PATIENCE num of iterations
            if len(monitor['val_acc']) > PATIENCE:
                # if the val acc hasnt improved in PATIENCE num of iterations
                if monitor['val_acc'][-PATIENCE] == np.max(monitor['val_acc'][-PATIENCE:]):
                    break

    with open(PATH+'data.txt', 'w') as outfile:
        json.dump(data, outfile)

    with open('latex.txt','w') as tex:
        line = '{} & {} & {} & {} & {} & {} \\\\\n'.format(
            lang_samples,len(loader.train_same),len(loader.train_diff),
            len(loader.test_same),len(loader.test_diff),np.max(monitor['val_acc'])
        )
        text.write(line)
