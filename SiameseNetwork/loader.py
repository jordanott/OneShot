import os
import json
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from keras.preprocessing.image import load_img

DATA_DIR = '../DataGeneration/'
class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self):
        self.info = {}
        self.data_same = []
        self.data_diff = []

        file_path = '../DataGeneration/data.json'
        print("loading data from {}".format(file_path))
        with open(file_path,"rb") as f:
            data = json.load(f)

        for i_key in data:
            # initialize language store
            for j_key in data:
                # generate images to sample
                m = len(data[i_key])
                i_index = np.random.randint(0,m,size=(70))
                m = len(data[j_key])
                j_index = np.random.randint(0,m,size=(70))
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

    def load_img_pair(self,pair):
        img1 = np.array(load_img(pair[0]))[:,:,0].reshape(400,400,1)
        img2 = np.array(load_img(pair[1]))[:,:,0].reshape(400,400,1)

        return img1,img2

    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        same_class = np.random.choice(len(self.data_same),batch_size/2)
        diff_class = np.random.choice(len(self.data_diff),batch_size/2)
        same_class = self.data_same[same_class]
        diff_class = self.data_diff[diff_class]

        pairs,targets = [],[]
        for pair in same_class:
            pairs.append(self.load_img_pair(pair))
            targets.append(1)
        for pair in diff_class:
            pairs.append(self.load_img_pair(pair))
            targets.append(0)

        return pairs, targets

    def make_oneshot_task(self,N,s="val",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        n_classes, n_examples, w, h = X.shape
        indices = rng.randint(0,n_examples,size=(N,))
        if language is not None:
            low, high = self.categories[s][language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = rng.choice(range(low,high),size=(N,),replace=False)

        else:#if no language specified just pick a bunch of random letters
            categories = rng.choice(range(n_classes),size=(N,),replace=False)
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
        test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
        support_set = X[categories,indices,:,:]
        support_set[0,:,:] = X[true_category,ex2]
        support_set = support_set.reshape(N, w, h,1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image,support_set]

        return pairs, targets

    def test_oneshot(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
