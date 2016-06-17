DATADIR = "TRAINING/"
MODELDIR = "MODELS/"

MODEL_ARCH_PREFIX = 'model_architecture__'
MODEL_WEIGHTS_PREFIX = 'model_weights__'


def load_training_data(geometry):

    import json
    import random

    filepaths = [ str(DATADIR + f + ".json") for f in geometry ]

    all_data = []

    for index in xrange(len(geometry)):

        filename = filepaths[index]
        geo = geometry[index]

        label = [0] * len(geometry)
        label[index] = 1

        with open(filename) as fp:
            data = json.load(fp)
            for datum in data:
                datum.update({'geometry':geo, 'label':label})

        
        all_data.extend(data)

    random.shuffle(all_data)

    return all_data

def split_data(data, split_ratio=0.2):

    split_index = int(len(data) * split_ratio)
    training_data = data[split_index:]
    test_data = data[:split_index]

    return (training_data, test_data)
 
