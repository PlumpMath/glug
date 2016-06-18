import GangGang
import Classify_helpers as ch

import subprocess
from functools import partial
from pprint import pprint

import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json


def load_model(model_suffix):

    ## LOAD MODEL
    model = model_from_json(open(ch.MODELDIR + ch.MODEL_ARCH_PREFIX + model_suffix + '.json').read())
    model.load_weights(ch.MODELDIR + ch.MODEL_WEIGHTS_PREFIX + model_suffix + '.h5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def predict(model, data):

    geometry = ["6poly", "tetrahedra", "cone", "cylinder"]

    ## PREDICT

    predict_data = np.array([data])

    classes = model.predict(predict_data)

    ## REPRESENT

    prediction = {}
    prediction['stats'] = dict(zip(geometry, classes[0].tolist()))

    predict_index = int(np.argmax(classes))
    prediction['prediction'] = geometry[predict_index]

    say_message = "predicted:" + geometry[predict_index]
    subprocess.Popen(["say", say_message])
    
    pprint(prediction)

    return prediction

if __name__ == "__main__":

    host = '172.16.15.1'
    port = 9093

    MODEL_SUFFIX = "6poly_tetra_cone_cyl__e200_b16"

    model = load_model(MODEL_SUFFIX)

    # 'preload' the predict function with a model by partial application
    predict_with_model = partial(predict, model)
    
    GangGang.server(host, port, predict_with_model)


