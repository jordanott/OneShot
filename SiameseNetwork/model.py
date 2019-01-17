from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Dropout, UpSampling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.applications.vgg16 import VGG16

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

def c_base():
    input_shape = (100, 100, 3)
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
    #convnet.add(Flatten())
    convnet.add(Dropout(.5))
    convnet.add(GlobalAveragePooling2D())
    convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3)))
    convnet.add(Dropout(.5))

    return convnet

def s_net(c=None):
    input_shape = (200, 200, 1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    #build convnet to use in each siamese 'leg'

    if c is not None:
        convnet = c
    else:
        convnet = c_base()

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

def c_net(num_languages=2,c=None):
    input_shape = (100, 100, 1)
    #build convnet to use in each siamese 'leg'
    if c is not None:
        convnet = c
    else:
        convnet = c_base()

    convnet.add(Dense(num_languages,activation='softmax'))

    optimizer = Adam(0.00006)
    convnet.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])

    return convnet

def load_from_ae(weights_path, SIAMESE=True):
    ae = ae_net()
    model = c_base()

    for i in range(len(ae.layers)):
        print model.layers[i].name
        if model.layers[i].name == 'last_conv':
            break
        params = ae.layers[i].get_weights()
        if params != []:
            model.layers[i].set_weights([params[0], params[1]])
    if SIAMESE: model = s_net(c=model)
    else: model = c_net(c=model)

    return model

def ae_net():
    input_shape = (200, 200, 1)
    #build convnet to use in each siamese 'leg'
    autoencoder = Sequential()

    autoencoder.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,kernel_regularizer=l2(2e-4), padding='same'))
    autoencoder.add(MaxPooling2D())
    autoencoder.add(Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2e-4), padding='same'))
    autoencoder.add(MaxPooling2D())
    autoencoder.add(Conv2D(128,(4,4),activation='relu',kernel_regularizer=l2(2e-4), padding='same'))
    autoencoder.add(MaxPooling2D())
    autoencoder.add(Conv2D(256,(4,4),activation='relu',kernel_regularizer=l2(2e-4), padding='same'))
    #autoencoder.add(MaxPooling2D())
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    autoencoder.add(Conv2D(256,(4,4),activation='relu',kernel_regularizer=l2(2e-4), padding='same'))
    autoencoder.add(UpSampling2D((2,2)))
    autoencoder.add(Conv2D(128,(4,4),activation='relu',kernel_regularizer=l2(2e-4), padding='same'))
    autoencoder.add(UpSampling2D((2,2)))
    autoencoder.add(Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2e-4), padding='same'))
    autoencoder.add(UpSampling2D((2,2)))
    autoencoder.add(Conv2D(1,(10,10),activation='sigmoid',kernel_regularizer=l2(2e-4), padding='same'))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

def VGG(input_shape,num_classes):
    input_tensor = Input(shape=input_shape)
    base_model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dense(1024, activation='relu')(x)
    # output layer with 'num_classes' number of catergories
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze pretrained layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    return model
