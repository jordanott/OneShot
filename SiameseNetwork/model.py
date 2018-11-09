from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Dropout, UpSampling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy

'''
import numpy.random as rng
def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)
'''
def s_net():
    input_shape = (200, 200, 1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    #build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                       kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(7,7),activation='relu',
                       kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(4,4),activation='relu',kernel_regularizer=l2(2e-4),name='last_conv'))
    convnet.add(Flatten())
    convnet.add(Dropout(.5))
    convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3)))
    convnet.add(Dropout(.5))
    #call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    return siamese_net

def c_net(num_languages=15):
    input_shape = (200, 200, 1)
    #build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                       kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(7,7),activation='relu',
                       kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(4,4),activation='relu',kernel_regularizer=l2(2e-4),name='last_conv'))
    convnet.add(Flatten())
    convnet.add(Dropout(.5))
    convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3)))
    convnet.add(Dropout(.5))
    convnet.add(Dense(num_languages,activation='softmax'))

    optimizer = Adam(0.00006)
    convnet.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])

    return convnet

def load_from_ae(weights_path, SIAMESE=True):
    ae = ae_net()
    if SIAMESE: model = s_net()
    else: model = c_net()

    for i in range(len(ae.layers)):
        if model.layers[i].name == 'last_conv':
            break
        params = ae.layers[i].get_weights()
        if params != []:
            model.layers[i].set_weights([params[0], params[1]])

    return model
    
def ae_net():
    input_shape = (200, 200, 1)
    #build convnet to use in each siamese 'leg'
    autoencoder = Sequential()
    autoencoder.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,kernel_regularizer=l2(2e-4)))
    autoencoder.add(MaxPooling2D())
    autoencoder.add(Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2e-4)))
    autoencoder.add(MaxPooling2D())
    autoencoder.add(Conv2D(128,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))
    autoencoder.add(MaxPooling2D())
    autoencoder.add(Conv2D(256,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))
    autoencoder.add(MaxPooling2D())
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    autoencoder.add(Conv2D(256,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))
    autoencoder.add(UpSampling2D())
    autoencoder.add(Conv2D(128,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))
    autoencoder.add(UpSampling2D())
    autoencoder.add(Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2e-4)))
    autoencoder.add(UpSampling2D())
    autoencoder.add(Conv2D(1,(10,10),activation='sigmoid',kernel_regularizer=l2(2e-4)))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder
