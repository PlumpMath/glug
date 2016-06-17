import Classify_helpers as ch

import sys, os
import random

import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

if len(sys.argv) < 2:
    print "ERROR: provide suffix for model file"
    exit(1)
MODEL_SUFFIX = sys.argv[1]

## LOAD DATA

geometry = ["6poly", "tetrahedra", "cone", "cylinder"]

all_data = ch.load_training_data(geometry)
datum = random.choice(all_data)

predict_data = np.array([datum['data']])
predict_label = datum['geometry']

print predict_data
print predict_data.shape
 
## LOAD MODULE

model = model_from_json(open(ch.MODELDIR + ch.MODEL_ARCH_PREFIX + MODEL_SUFFIX + '.json').read())
model.load_weights(ch.MODELDIR + ch.MODEL_WEIGHTS_PREFIX + MODEL_SUFFIX + '.h5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

## PREDICT

classes = model.predict(predict_data)

## REPRESENT

for g in zip(geometry, classes[0]):
    print g[0], ":", "{:.8f}".format(g[1])
print ""

predict_index = np.argmax(classes)
message =  "predicted:" + geometry[predict_index]

print message
os.system("say " + message)

