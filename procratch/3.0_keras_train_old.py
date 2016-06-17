import os
import helpers
import random
import sys

#import matplotlib.pyplot as plot
import numpy as np
import re
from functools import partial
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

if len(sys.argv) < 2:
    print "ERROR: provide suffix for model file"
    exit(1)
MODEL_SUFFIX = sys.argv[1]

LOAD_RAWS = False

######### get training paths

trainingdir = None
file_to_array = None

if(LOAD_RAWS):
    trainingdir = helpers.RAW_TRAINING_DIR
    file_to_array = partial(helpers.file_to_array, reduce=True)
    print "loading RAW training images...."
else:
    trainingdir = helpers.REDUCED_TRAINING_DIR
    file_to_array = partial(helpers.file_to_array, reduce=False)
    print "loading REDUCED training images...."


training_paths = []
for path, subdirs, files in os.walk(trainingdir):
    for name in files:
        fullpath = os.path.join(path, name)
        if not re.match(r".*DS_Store", fullpath):
            training_paths.append(fullpath)


######### generate training data
random.shuffle(training_paths)


training_data = np.array(map(file_to_array, training_paths))
training_data = np.reshape(training_data, (-1, 3000))
training_labels = np.array(map(helpers.file_to_label, training_paths))

print training_data.shape


print map(lambda x: x.shape, training_data)


#print training_data
#print training_labels

######## sMODEL

model = Sequential()

model.add(Dense(input_dim=3000, output_dim=200, input_shape=(3, 25, 40)))
model.add(Activation("tanh"))

model.add(Dense(input_dim=100, output_dim=50))
model.add(Activation("tanh"))

model.add(Dense(input_dim=50, output_dim=20))
model.add(Activation("tanh"))

model.add(Dense(input_dim=20, output_dim=1))
model.add(Activation("sigmoid"))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

print("begin to train")

model.fit(training_data, training_labels, nb_epoch = 3000, batch_size=6, verbose=1, shuffle=True)

####### SAVING

print ("saving model to file..")

json_string = model.to_json()
open(helpers.MODELDIR + 'model_architecture__' + MODEL_SUFFIX + '.json', 'w').write(json_string)
model.save_weights(helpers.MODELDIR + 'model_weights__' + MODEL_SUFFIX + '.h5')


"""
test = np.array(([0,1], [1,1]))
classes = model.predict(test)
print classes
"""





