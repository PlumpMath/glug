import Classify_helpers as ch
import numpy as np
import sys

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

if len(sys.argv) < 2:
    print "ERROR: provide suffix for model file"
    exit(1)
MODEL_SUFFIX = sys.argv[1]



def save_model_to_file(model, MODEL_SUFFIX):

    json_string = model.to_json()
    open(ch.MODELDIR + ch.MODEL_ARCH_PREFIX + MODEL_SUFFIX + '.json', 'w').write(json_string)
    model.save_weights(ch.MODELDIR + ch.MODEL_WEIGHTS_PREFIX + MODEL_SUFFIX + '.h5')


def makeCNN():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ch.IMAGE_HEIGHT, ch.IMAGE_WIDTH)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
     
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, ch.IMAGE_HEIGHT, ch.IMAGE_WIDTH)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
     
    model.add(Flatten())
    model.add(Dense(input_dim = (64*8*8), output_dim = 512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
     
    model.add(Dense(input_dim=512, output_dim=2))
    model.add(Activation('softmax'))

    model = makeCNN()
    return model


def makeNormal():
    model = Sequential()
    
    model.add(Dense(500, input_shape=(1000,), init='uniform'))
    model.add(Activation("tanh"))
    model.add(Dropout(0.25))

    model.add(Dense(500, init='uniform'))
    model.add(Activation("tanh"))
    model.add(Dropout(0.25))

    model.add(Dense(50, init = 'uniform'))
    model.add(Activation("tanh"))
    model.add(Dropout(0.5))

    model.add(Dense(len(geometry)))
    model.add(Activation("softmax"))
    return model

def make_hera_model():
    from heraspy.model import HeraModel

    hera_model = HeraModel(
        {
            'id': 'my-model' # any ID you want to use to identify your model
        },
        {
            # location of the local hera server, out of the box it's the following
            'domain': 'localhost',
            'port': 4000
        }
    )
    return hera_model

#####################
#####################

if __name__ == '__main__':

    geometry = ["6poly", "tetrahedra", "cone", "cylinder"]

    all_data = ch.load_training_data(geometry)

    (training_data, test_data) = ch.split_data(all_data, 0.2)

    training_data_np = np.array(map(lambda x: x['data'], training_data))
    training_labels_np = np.array(map(lambda x: x['label'], training_data))

    test_data_np = np.array(map(lambda x: x['data'], test_data))
    test_labels_np = np.array(map(lambda x: x['label'], test_data))

    model = makeNormal()

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    #model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    #model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    hera_model = make_hera_model()


    print("begin to train")

    history = model.fit(training_data_np, training_labels_np,
            nb_epoch = 200, 
            batch_size= 16, 
            verbose= 1, 
            validation_split=0.2,
            shuffle=True,
            callbacks=[hera_model.callback])



    score = model.evaluate(test_data_np, test_labels_np)
   
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    ####### SAVING

    print ("saving model to file..")

    save_model_to_file(model, MODEL_SUFFIX)


