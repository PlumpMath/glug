import time
import re
import os
import helpers
import random
import PIL
from PIL import Image
#import matplotlib.pyplot as plot
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import sys

if len(sys.argv) < 2:
    print "ERROR: provide suffix for model file"
    exit(1)
MODEL_SUFFIX = sys.argv[1]

while(True):

    ## TAKE SCREENSHOT 
    print "Taking screenshot in..."
    for i in xrange(3, 0, -1):
        time.sleep(0.5)
        print "Taking screenshot in...", i
    print "Taking screenshot in... NOW!"

    timestr = time.strftime("%Y%m%d-%H%M%S")
    timestr = "now"
    filename = helpers.TESTDIR + "screencap-" + timestr + ".png"
    os.system("screencapture -x " + filename)

    ## LOAD IMAGE

    img = image.load_img(filename)
    img = img.resize((helpers.IMAGE_WIDTH, helpers.IMAGE_HEIGHT), PIL.Image.ANTIALIAS)
    imgarr = np.array(image.img_to_array(img))
    imgarr = np.reshape(imgarr, (-1, helpers.ARRAY_DIM))

    ## LOAD MODULE

    model = model_from_json(open(helpers.MODELDIR + 'model_architecture__' + MODEL_SUFFIX + '.json').read())
    model.load_weights(helpers.MODELDIR + 'model_weights__' + MODEL_SUFFIX + '.h5')
    model.compile(optimizer='adagrad', loss='mse')

    classes = model.predict(imgarr)
    chance = classes[0][0]

    print "chance that you are procrastinating right now:"
    print chance, "%, or:",
    textchance = helpers.prob_to_text(chance)
    print textchance

    if(chance > 0.5):
        voice = random.choice(helpers.allvoices)
        os.system("say -v " + voice + " You are " + textchance)

    time.sleep(3)

