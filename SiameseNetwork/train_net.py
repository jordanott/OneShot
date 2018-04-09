from loader import Siamese_Loader
from model import siamese_net
import matplotlib.pyplot as plt
import numpy as np
import json
import os

PATIENCE = 20
batch_size = 16
n_iter = 1000000
best = -1

with open('latex.txt','w') as tex:
    line = 'Language Samples & Train Same & Train Diff & Test Same & Test Diff & Val Acc & Batches \\\\\n'
    tex.write(line)

for lang_samples in range(1,132,10):
    PATH = str(lang_samples)+'/'
    os.mkdir(PATH)
    evaluate_every = lang_samples*10
    weights_path = PATH + 'weights.h5'

    train_loader = Siamese_Loader(lang_samples)
    test_loader = Siamese_Loader(lang_samples)
    monitor = {
        'train_acc': [],
        'val_acc': []
    }
    tmp_train_acc = []

    for i in range(1, n_iter):
        (inputs,targets) = train_loader.get_batch(batch_size)
        history = siamese_net.fit(inputs,targets,verbose=0)

        tmp_train_acc.append(history.history['acc'][-1])

        if i % evaluate_every == 0:
            val_acc = test_loader.test_oneshot(siamese_net,batch_size,verbose=True)
            print 'Samples',lang_samples,'Batches:',i,'Validation accuracy:',val_acc
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

    with open(PATH+'data.json', 'w') as outfile:
        json.dump(monitor, outfile)

    with open('latex.txt','a') as tex:
        line = '{} & {} & {} & {} & {} & {} & {} \\\\\n'.format(
            lang_samples,len(train_loader.train_same),len(train_loader.train_diff),
            len(train_loader.test_same),len(train_loader.test_diff),np.max(monitor['val_acc']),i
        )
        tex.write(line)
