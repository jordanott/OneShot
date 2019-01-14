import os
import json
import keras
import random
import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.utils import shuffle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(0)

DATA_DIR = '../pngPML_Jan10/'

datagen = ImageDataGenerator(width_shift_range=0.1,
                    height_shift_range=0.1,
                    #shear_range=0.1,
                    zoom_range=0.2,
                    horizontal_flip=False)

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,lang_samples,test_samples_per_lang=100):
        self.languages = {}
        self.data_same = []
        self.data_diff = []

        #file_path = '../DataGeneration/data.json'
        df = pd.read_csv('../png_uml.csv')
        data = {
            'Class':df[df['CHART_TYPE'] == 'Class']['NAME'].values.tolist(),
            'Sequence':df[df['CHART_TYPE'] == 'Sequence']['NAME'].values.tolist()
        }
        #with open(file_path,"rb") as f:
        #    data = json.load(f)

        print df.shape
        '''
        err = 0
        for index, row in df.iterrows():
            f = row['NAME']
            try:
                shape = np.array(load_img(DATA_DIR+f)).shape
                if shape[0] < 100 or shape[1] < 100:
                    df.drop(index, inplace=True)
            except:
                err += 1
                df.drop(index, inplace=True)

        print df.shape
        df.to_csv('../new.csv',index=False)
        print err
        '''


        for i_key in data: # { lang_name: file_names_for_images}
            self.languages[i_key] = []
            # num samples for that lang
            m = len(data[i_key])
            # images that will be used for testing
            test_index = np.random.randint(0,m,size=(test_samples_per_lang))

            for t in test_index: self.languages[i_key].append(DATA_DIR+data[i_key][t])

            # remove images that will go in the test set
            test_index = sorted(test_index, reverse=True)
            for t in test_index: del data[i_key][t]

            for j_key in data:
                # generate images to sample
                m = len(data[i_key])
                i_index = np.random.randint(0,m,size=(lang_samples))
                m = len(data[j_key])
                j_index = np.random.randint(0,m,size=(lang_samples))
                # iterate through sampled images and pair samples
                for i in i_index:
                    for j in j_index:
                        # if the languages are the same; put in same store
                        if i_key == j_key:
                            self.data_same.append([DATA_DIR+data[i_key][i], DATA_DIR+data[j_key][j]])
                        else:
                            self.data_diff.append([DATA_DIR+data[i_key][i], DATA_DIR+data[j_key][j]])
        self.data_same = np.array(self.data_same)
        self.data_diff = np.array(self.data_diff)

        self.fold_reset()

    def fold_reset(self):
        np.random.shuffle(self.data_same)
        np.random.shuffle(self.data_diff)

        self.len_test = 0
        for k in self.languages:
            self.len_test += len(self.languages[k])

        print 'Training set'
        print '\tSame:',len(self.data_same),'Diff:',len(self.data_diff)

        print 'Testing set'
        print '\tLength:',self.len_test

    def load_img_pair(self,pair):
        print np.array(load_img(pair[0])).shape, pair[0]
        img1 = np.array(load_img(pair[0]))[:,:,0]#.reshape(200,200,1)
        img2 = np.array(load_img(pair[1]))[:,:,0]#.reshape(200,200,1)
        return img1/255.0,img2/255.0

    def get_batch(self,batch_size):
        """Create batch of n pairs, half same class, half different class"""
        same = self.data_same
        diff = self.data_diff

        same_class = np.random.choice(len(same),batch_size/2)
        diff_class = np.random.choice(len(diff),batch_size/2)
        same_class = same[same_class]
        diff_class = diff[diff_class]

        pairs,targets = [[],[]],[]
        for pair in same_class:
            img1,img2 = self.load_img_pair(pair)
            pairs[0].append(datagen.random_transform(img1))
            pairs[1].append(datagen.random_transform(img2))
            targets.append(1)
        for pair in diff_class:
            img1,img2 = self.load_img_pair(pair)
            pairs[0].append(datagen.random_transform(img1))
            pairs[1].append(datagen.random_transform(img2))
            targets.append(0)

        return [np.array(pairs[0]),np.array(pairs[1])], targets

    def pick_diff_language(self, not_this_lang):
        lang = np.random.choice(self.languages.keys())
        if lang == not_this_lang:
            return self.pick_diff_language(not_this_lang)
        return lang

    def test_batch(self,batch_size):
        pairs, targets = [[],[]], []
        # same language
        for i in range(batch_size / 2):
            # chose a language
            lang = np.random.choice(self.languages.keys())
            # chose two images from the same language
            test_img = np.random.choice(self.languages[lang])
            other = np.random.choice(self.languages[lang])

            img1, img2 = self.load_img_pair([DATA_DIR+test_img, DATA_DIR+other])
            pairs[0].append(img1);pairs[1].append(img2)
            targets.append(1)
        # different language
        for i in range(batch_size / 2):
            # chose a language and image from lang
            lang = np.random.choice(self.languages.keys())
            test_img = np.random.choice(self.languages[lang])
            # chose image from a different lang
            diff_lang = self.pick_diff_language(lang)
            other = np.random.choice(self.languages[diff_lang])

            img1, img2 = self.load_img_pair([DATA_DIR+test_img, DATA_DIR+other])
            pairs[0].append(img1); pairs[1].append(img2)
            targets.append(0)

        return [np.array(pairs[0]),np.array(pairs[1])], targets

    def test_oneshot(self,model,batch_size,s="val",verbose=0):
        correct = 0
        for b in range(0,self.len_test):
            inputs, targets = self.test_batch(batch_size)
            probs = model.predict(inputs)
            pred = (probs > .5).flatten()
            correct += np.sum( targets == pred ) / float(len(probs))
        return correct/float(self.len_test)

class Conv_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,lang_samples,test_samples_per_lang=100):
        self.x_train, self.y_train = [], []
        self.x_test, self.y_test = [], []

        '''file_path = '../DataGeneration/data.json'
        print("loading data from {}".format(file_path))
        with open(file_path,"rb") as f:
            data = json.load(f)
        '''

        df = pd.read_csv('../new.csv')
        data = {
            'Class':df[df['CHART_TYPE'] == 'Class']['NAME'].values.tolist(),
            'Sequence':df[df['CHART_TYPE'] == 'Sequence']['NAME'].values.tolist()
        }
        #with open(file_path,"rb") as f:
        #    data = json.load(f)

        print df.shape

        lang_counter = 0
        for i_key in data:
            m = len(data[i_key])
            test_index = np.random.randint(0,m,size=(test_samples_per_lang))
            for t in test_index:
                self.x_test.append(DATA_DIR+data[i_key][t])
                self.y_test.append(lang_counter)

            # remove images that will go in the test set
            test_index = sorted(test_index, reverse=True)
            for t in test_index: del data[i_key][t]

            m = len(data[i_key])
            train_index = np.random.randint(0,m,size=(lang_samples))
            for t in train_index:
                self.x_train.append(DATA_DIR+data[i_key][t])
                self.y_train.append(lang_counter)

            lang_counter += 1

        self.x_train = np.array(self.x_train)
        self.y_train = keras.utils.to_categorical(self.y_train, lang_counter)

        self.x_test = np.array(self.x_test)
        self.y_test = keras.utils.to_categorical(self.y_test, lang_counter)

        self.fold_reset()

    def shuffle(self,x,y):
        c = list(zip(x, y))
        random.shuffle(c)
        a, b = zip(*c)
        return np.array(a),np.array(b)

    def fold_reset(self):
        self.x_train,self.y_train = self.shuffle(self.x_train,self.y_train)
        self.x_test,self.y_test = self.shuffle(self.x_test,self.y_test)

        self.len_test = len(self.x_test)

        print 'Training set'
        print '\tLength:',self.x_train.shape

        print 'Testing set'
        print '\tLength:',self.len_test

    def load_img(self,img_location):

        img = np.array(load_img(img_location))
        if img.shape[0] > 200 and img.shape[1] > 200:
            img = img[::2,::2,0][:100,:100].reshape(100,100,1)
        else:
            img = img[:100,:100,0].reshape(100,100,1)
        return img/255.0

    def get_batch(self,batch_size):
        train_idx = np.random.choice(len(self.x_train),batch_size)
        locations = self.x_train[train_idx]
        languages = self.y_train[train_idx]

        data,targets = [],[]
        for loc,lang in zip(locations,languages):
            img1 = self.load_img(loc)
            data.append(img1)
            targets.append(lang)

        return np.array(data), np.array(targets)

    def test_batch(self,batch_size):
        test_idx = np.random.choice(len(self.x_test),batch_size)
        locations = self.x_test[test_idx]
        languages = self.y_test[test_idx]

        data,targets = [],[]
        for loc,lang in zip(locations,languages):
            img1 = self.load_img(loc)
            data.append(datagen.random_transform(img1))
            targets.append(lang)

        return np.array(data), np.array(targets)

    def test_oneshot(self,model,batch_size,verbose=0):
        correct = 0
        for b in range(0,self.len_test):
            inputs, targets = self.test_batch(batch_size)
            probs = model.predict(inputs)
            correct += np.sum( np.argmax(targets,axis=1) == np.argmax(probs,axis=1) ) / float(len(probs))
        return correct/float(self.len_test)
