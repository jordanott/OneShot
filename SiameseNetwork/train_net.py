from loader import Siamese_Loader, Conv_Loader
from model import s_net, c_net
import matplotlib.pyplot as plt
import numpy as np
import json
import os

SIAMESE = False
PATIENCE = 20
batch_size = 16
n_iter = 1000000
best = -1

if SIAMESE:
    with open('s_latex.txt','w') as tex:
        line = 'Language Samples & Train Same & Train Diff & Test & Val Acc & Batches \\\\\n'
        tex.write(line)
    net = s_net()
    net.save_weights('s_weights.h5')
    weights = 's_weights.h5'
    results = 'S_Results/'
else:
    with open('c_latex.txt','w') as tex:
        line = 'Language Samples & Train & Test & Val Acc & Batches \\\\\n'
        tex.write(line)
    net = c_net()
    net.save_weights('c_weights.h5')
    weights = 'c_weights.h5'
    results = 'C_Results/'

if not os.path.exists(results):
    os.mkdir(results)

for lang_samples in range(1,1002,50):
    PATH = results+str(lang_samples)+'/'
    if not os.path.exists(PATH):    
        os.mkdir(PATH)
    evaluate_every = 100
    weights_path = PATH + weights

    if SIAMESE:
        loader = Siamese_Loader(lang_samples)
    else:
        loader = Conv_Loader(lang_samples)
    monitor = {
        'train_acc': [],
        'val_acc': []
    }
    tmp_train_acc = []
    # use same initialization for all trials
    net.load_weights(weights)

    for i in range(1, n_iter):
        # get batch of data; images and labels
        (inputs,targets) = loader.get_batch(batch_size)
        history = net.fit(inputs,targets,verbose=0)

        tmp_train_acc.append(history.history['acc'][-1])

        if i % evaluate_every == 0:
            # evaluate network on test
            val_acc = loader.test_oneshot(net,batch_size,verbose=True)
        
            monitor['val_acc'].append(val_acc)
            if val_acc >= best:
                #print("saving")
                net.save(weights_path)
                best=val_acc

            # mean of train accuracies
            train_acc = np.mean(tmp_train_acc)
            tmp_train_acc = []
            monitor['train_acc'].append(train_acc)

            print("Samples: {}, Iteration: {}, Avg Training Acc: {:.2f}, Val Acc {:.3f}".format(
                lang_samples,i//evaluate_every,train_acc, val_acc))

            # if there has been at least PATIENCE num of iterations
            if len(monitor['val_acc']) > PATIENCE:
                # if the val acc hasnt improved in PATIENCE num of iterations
                if monitor['val_acc'][-PATIENCE] == np.max(monitor['val_acc'][-PATIENCE:]):
                    break

    with open(PATH+'data.json', 'w') as outfile:
        json.dump(monitor, outfile)

    if SIAMESE:
        with open('s_latex.txt','a') as tex:
            line = '{} & {} & {} & {} & {} & {} \\\\\n'.format(
                lang_samples,len(loader.data_same),len(loader.data_diff),
                loader.len_test,np.max(monitor['val_acc']),i
            )
            tex.write(line)
    else:
        with open('c_latex.txt','a') as tex:
            line = '{} & {} & {} & {} & {} \\\\\n'.format(
                lang_samples,len(loader.x_train),
                loader.len_test,np.max(monitor['val_acc']),i
            )
            tex.write(line)
