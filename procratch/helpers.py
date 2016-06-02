##DIRECTORY STUFF

TRAINING_DIR = "TRAINING/"
RAW_SUFFIX = "RAW/"
REDUCED_SUFFIX = "REDUCED/"

RAW_TRAINING_DIR = TRAINING_DIR + RAW_SUFFIX
REDUCED_TRAINING_DIR = TRAINING_DIR + REDUCED_SUFFIX

MODELDIR = "MODELS/"

TESTDIR = "TESTCAPTURES/"

## IMAGE DIMENSIONS
IMAGE_WIDTH = 40 
IMAGE_HEIGHT = 25 

_slacking_text = ["SUPER LOW", "LOW", "KINDA LOW", "NOT SURE", "GETTING OFF TRACK", "PROBABLY PROCRASTINATING", "TOTALLY SLACKING"]
_slacking_probability = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

allvoices = ["Agnes", "Albert", "Alex", "Alice", "Alva", "Amelie", "Anna", "Ava", "Bad News", "Bahh", "Bells", "Boing", "Bruce", "Bubbles", "Carmit", "Cellos", "Damayanti", "Daniel", "Deranged", "Diego", "Ellen", "Fiona", "Fred", "Good News", "Hysterical", "Ioana", "Joana", "Junior", "Kanya", "Karen", "Kathy", "Kyoko", "Laura", "Lekha", "Luciana", "Maged", "Mariska", "Mei-Jia", "Melina", "Milena", "Moira", "Monica", "Nora", "Paulina", "Pipe Organ", "Princess", "Ralph", "Samantha", "Sara", "Satu", "Sin-ji", "Tessa", "Thomas", "Ting-Ting", "Trinoids", "Veena", "Vicki", "Victoria", "Whisper", "Xander", "Yelda", "Yuna", "Zarvox", "Zosia", "Zuzana"]


def prob_to_text(prob):
    for i in xrange(len(_slacking_probability)):
        if _slacking_probability[i] > prob:
            return _slacking_text[i]
    return _slacking_text[-1]


def file_to_array(filename, reduce=True):
    from keras.preprocessing import image as kerasimage
    import numpy as np
    import PIL

    loaded_img = kerasimage.load_img(filename)

    if (reduce):
        img = loaded_img.resize([IMAGE_WIDTH, IMAGE_HEIGHT],PIL.Image.ANTIALIAS)
    else:
        img = loaded_img

    imgarr = kerasimage.img_to_array(img)

    nparr = np.asarray(imgarr)

    return np.reshape(nparr, (-1, IMAGE_WIDTH * IMAGE_HEIGHT * 3))

def file_to_label(filename):
    import re
    if re.match(r".*YES.*", filename):
        return [1]
    else:
        return [0]


