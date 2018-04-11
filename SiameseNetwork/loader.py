import os
import json
import random
import numpy as np
#import seaborn as sns
from sklearn.utils import shuffle
from keras.preprocessing.image import load_img

DATA_DIR = '../DataGeneration/'
class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,lang_samples):
        self.languages = {}
        self.data_same = []
        self.data_diff = []

        file_path = '../DataGeneration/data.json'
        print("loading data from {}".format(file_path))
        with open(file_path,"rb") as f:
            data = json.load(f)

        for i_key in data:
            self.languages[i_key] = data[i_key]
            for j_key in data:
                # generate images to sample
                m = len(data[i_key])
                i_index = np.random.randint(0,m,size=(lang_samples))
                m = len(data[j_key])
                j_index = np.random.randint(0,m,size=(lang_samples))
                # iterate through sampled images and pair samples
                for i in i_index:
                    for j in j_index:
                        label = 0 if i_key != j_key else 1
                        if label:
                            self.data_same.append([DATA_DIR+data[i_key][i], DATA_DIR+data[j_key][j]])
                        else:
                            self.data_diff.append([DATA_DIR+data[i_key][i], DATA_DIR+data[j_key][j]])
        self.data_same = np.array(self.data_same)
        self.data_diff = np.array(self.data_diff)

        self.fold_reset()

    def fold_reset(self):
        np.random.shuffle(self.data_same)
        np.random.shuffle(self.data_diff)

        self.train_same = self.data_same[:int(.8*len(self.data_same))]
        self.train_diff = self.data_diff[:int(.8*len(self.data_diff))]

        self.test_same = self.data_same[int(.8*len(self.data_same)):]
        self.test_diff = self.data_diff[int(.8*len(self.data_diff)):]

        print 'Training set'
        print '\tSame:',len(self.train_same),'Diff:',len(self.train_diff)

        print 'Testing set'
        print '\tSame:',len(self.test_same),'Diff:',len(self.test_diff)

    def load_img_pair(self,pair):
        img1 = np.array(load_img(pair[0]))[:,:,0].reshape(200,200,1)
        img2 = np.array(load_img(pair[1]))[:,:,0].reshape(200,200,1)
        return img1/255.0,img2/255.0

    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        if s == 'train':
            same = self.train_same
            diff = self.train_diff
        else:
            same = self.test_same
            diff = self.test_diff

        same_class = np.random.choice(len(same),batch_size/2)
        diff_class = np.random.choice(len(diff),batch_size/2)
        same_class = same[same_class]
        diff_class = diff[diff_class]

        pairs,targets = [[],[]],[]
        for pair in same_class:
            img1,img2 = self.load_img_pair(pair)
            pairs[0].append(img1)
            pairs[1].append(img2)
            targets.append(1)
        for pair in diff_class:
            img1,img2 = self.load_img_pair(pair)
            pairs[0].append(img1)
            pairs[1].append(img2)
            targets.append(0)

        if s != 'test':
            return [np.array(pairs[0]),np.array(pairs[1])], targets
        else:
            return [np.array(pairs[0]),np.array(pairs[1])], targets, np.concatenate((same_class,diff_class),axis=0)

    def test_batch(self):
        lang = np.random.choice(self.languages.keys())
        test_img = np.random.choice(self.languages[lang])
        pairs, targets = [[],[]], []
        for lang_key in self.languages.keys():
            other = np.random.choice(self.languages[lang_key])
            img1, img2 = self.load_img_pair([DATA_DIR+test_img, DATA_DIR+other])
            pairs[0].append(img1)
            pairs[1].append(img2)
            l = 1 if lang == lang_key else 0
            targets.append(l)
        return [np.array(pairs[0]),np.array(pairs[1])], targets

    def test_oneshot(self,model,batch_size,s="val",verbose=0):
        correct = 0
        test_len = len(self.test_same) + len(self.test_diff)
        total = 0
        for b in range(0,test_len,batch_size*10):
            inputs, targets = self.test_batch()
            probs = model.predict(inputs)
            correct += 1 if np.argmax(probs) == np.argmax(targets) else 0
            total += 1
        return correct/float(total)
