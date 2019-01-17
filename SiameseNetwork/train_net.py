from loader import Siamese_Loader, Conv_Loader
from model import s_net, c_net, load_from_ae, VGG

import matplotlib.pyplot as plt
import numpy as np
import json
import os

RESULTS_DIR='Shallow/' 
RESULTS_DIR='VGG_PreTrained/'

LOAD_FROM_AE = False
SIAMESE = False
PATIENCE = 5
batch_size = 32
n_iter = 10000

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
    with open(RESULTS_DIR+'latex.txt','w') as tex:
        line = 'Num Samples & Train & Test & Val Acc & Epochs\\\\\n'
        tex.write(line)

    if 'VGG' in RESULTS_DIR: net = VGG((100,100,3),2)
    else: net = c_net() #if not LOAD_FROM_AE else load_from_ae('ae.h5',SIAMESE)

    net.save_weights(RESULTS_DIR+'weights.h5')
    weights = RESULTS_DIR+'weights.h5'
    results = RESULTS_DIR+'Results/'

if not os.path.exists(results):
    os.mkdir(results)

for lang_samples in range(50,2000,250):
    PATH = results+str(lang_samples)+'/'
    if not os.path.exists(PATH): os.mkdir(PATH)

    fold_info = {'fold_val_acc':[], 'epoch':[]}
    fold = 0
    while fold < 10:
        print 'Fold:', fold

        FOLD_PATH = PATH+str(fold)+'/'
        if not os.path.exists(FOLD_PATH): os.mkdir(FOLD_PATH)

        weights_path = FOLD_PATH +'weights.h5'

        if SIAMESE:
            loader = Siamese_Loader(lang_samples)
        else:
            loader = Conv_Loader(lang_samples)
        monitor = {'train_acc': [],'val_acc': [],'p':[],'r':[],'f':[]}

        # use same initialization for all trials
        net.load_weights(weights)
        (inputs,targets) = loader.get_batch()
        print np.sum(np.argmax(targets,axis=1)), len(targets)


        '''for i in range(len(targets)):
            plt.imshow(inputs[i])
            plt.savefig(str(i)+'_'+str(np.argmax(targets[i])) )
        '''
        for epoch in range(1, n_iter):
            
            history = net.fit(inputs,targets,verbose=0)

            # evaluate network on test
            val_acc,p,r,f,test_percent = loader.test_oneshot(net,verbose=True)

            monitor['val_acc'].append(val_acc)
            monitor['p'].append(p);monitor['r'].append(r);monitor['f'].append(f)

            # checking for best validation acc
            if val_acc >= np.max(monitor['val_acc']): net.save(weights_path)

            # mean of train accuracies
            monitor['train_acc'].append(history.history['acc'][-1])

            print("Samples: {}, Iteration: {}, Avg Training Acc: {:.2f}, Val Acc {:.3f}, Test % {:.3f}, P {:.3f}, R {:.3f}, F {:.3f}".format(
                lang_samples,epoch,monitor['train_acc'][-1], val_acc, test_percent,p,r,f))

            # if there has been at least PATIENCE num of iterations
            if len(monitor['val_acc']) > PATIENCE:
                # if the val acc hasnt improved in PATIENCE num of iterations
                if monitor['val_acc'][-PATIENCE] == np.max(monitor['val_acc'][-PATIENCE:]):                    
                    if np.max(monitor['val_acc']) == .5 and 'Shallow' in RESULTS_DIR:
                        fold -= 1
                    else:
                        fold_info['fold_val_acc'].append(np.max(monitor['val_acc']))
                        fold_info['epoch'].append(epoch)
                    break
        fold += 1      

    with open(FOLD_PATH+'data.json', 'w') as outfile:
        json.dump(monitor, outfile)

    with open(RESULTS_DIR+'latex.txt','a') as tex:
        line = '{} & {} & {} & {} & {}\\\\\n'.format(
            lang_samples,len(loader.x_train),
            loader.len_test,np.mean(fold_info['fold_val_acc']),np.mean(fold_info['epoch'])
        )
        tex.write(line)
