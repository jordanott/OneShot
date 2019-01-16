from loader import Siamese_Loader, Conv_Loader
from model import s_net, c_net, load_from_ae, VGG

import matplotlib.pyplot as plt
import numpy as np
import json
import os

RESULTS_DIR='Shallow/' #'VGG_PreTrained/'
RESULTS_DIR='VGG/'

LOAD_FROM_AE = False
SIAMESE = False
PATIENCE = 5
batch_size = 32
n_iter = 1000000
best = -1

np.random.seed(0)

if SIAMESE:
    with open('s_latex.txt','w') as tex:
        line = 'Language Samples & Train Same & Train Diff & Test & Val Acc & Batches \\\\\n'
        tex.write(line)
    net = s_net() if not LOAD_FROM_AE else load_from_ae('ae.h5',SIAMESE)
    net.save_weights('s_weights.h5')
    weights = 's_weights.h5'
    results = 'S_Results/'
else:
    with open(RESULTS_DIR+'c_latex.txt','w') as tex:
        line = 'Language Samples & Train & Test & Val Acc & Batches P & R & F\\\\\n'
        tex.write(line)

    if 'VGG' in RESULTS_DIR: net = VGG((100,100,3),2) 
    else: net = c_net() #if not LOAD_FROM_AE else load_from_ae('ae.h5',SIAMESE)

    net.save_weights(RESULTS_DIR+'c_weights.h5')
    weights = RESULTS_DIR+'c_weights.h5'
    results = RESULTS_DIR+'C_Results/'

if not os.path.exists(results):
    os.mkdir(results)

for lang_samples in range(50,2000,250):
    PATH = results+str(lang_samples)+'/'
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    evaluate_every = 100
    weights_path = PATH +'weights.h5'

    if SIAMESE:
        loader = Siamese_Loader(lang_samples)
    else:
        loader = Conv_Loader(lang_samples)
    monitor = {
        'train_acc': [],
        'val_acc': [],
        'p':[],
        'r':[],
        'f':[]
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
            val_acc,p,r,f,test_percent = loader.test_oneshot(net,batch_size,verbose=True)
            
            monitor['val_acc'].append(val_acc)
            monitor['p'].append(p);monitor['r'].append(r);monitor['f'].append(f)
            if val_acc >= best:
                #print("saving")
                net.save(weights_path)
                best=val_acc

            # mean of train accuracies
            train_acc = np.mean(tmp_train_acc)
            tmp_train_acc = []
            monitor['train_acc'].append(train_acc)

            print("Samples: {}, Iteration: {}, Avg Training Acc: {:.2f}, Val Acc {:.3f}, Test % {:.3f}, P {:.3f}, R {:.3f}, F {:.3f}".format(
                lang_samples,i//evaluate_every,train_acc, val_acc, test_percent,p,r,f))

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
        with open(RESULTS_DIR+'c_latex.txt','a') as tex:
            line = '{} & {} & {} & {} & {} & {} & {} & {}\\\\\n'.format(
                lang_samples,len(loader.x_train),
                loader.len_test,np.max(monitor['val_acc']),i, np.max(monitor['p']),np.max(monitor['r']),np.max(monitor['f'])
            )
            tex.write(line)
