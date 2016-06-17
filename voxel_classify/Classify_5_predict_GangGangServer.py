import socket
import pickle
import time
import numpy as np

def recv_timeout(the_socket , timeout=1):
    the_socket.setblocking(0)
     
    total_data=[]
    data=''
     
    begin = time.time()

    while True:
        #if you got some data, then break after timeout
        if total_data and time.time() - begin > timeout:
            break
        #if you got no data at all, wait a little longer, twice the timeout
        elif time.time() - begin > timeout * 2:
            break
        #recv something
        try:
            data = the_socket.recv(8192)
            if data:
                total_data.append(data)
                #change the beginning time for measurement
                begin=time.time()
            else:
                #sleep for sometime to indicate a gap
                time.sleep(0.1)
        except:
            pass
    #join all parts to make final string
    return ''.join(total_data)

def recv_unpickle(socket, custom_function):
    data = recv_timeout(socket)
    if len(data) > 0:
        try:
            process_data( pickle.loads(data) , socket, custom_function)
        except EOFError, e:
            return None

def process_data(data, socket, custom_function):

    if type(data).__name__ != 'list':
        raise TypeError('data is not a list!') 

    result = custom_function(data)
    
    socket.sendall(str(result))


def listen_and_execute(host, port, custom_function):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((host, port))
    serversocket.listen(5) # become a server socket, maximum 5 connections

    while True:
        conn, addr = serversocket.accept()
        recv_unpickle(conn, custom_function)

############
############
############


def sumsum(data):
    import Classify_helpers as ch

    import sys, os
    import random

    import numpy as np
    from keras.preprocessing import image
    from keras.models import model_from_json

    MODEL_SUFFIX = "6poly_tetra_cone_cyl__e200_b16"
    
    ## LOAD DATA

    geometry = ["6poly", "tetrahedra", "cone", "cylinder"]

    predict_data = np.array([data])

    print predict_data
    print predict_data.shape
    

    ## LOAD MODULE

    model = model_from_json(open(ch.MODELDIR + ch.MODEL_ARCH_PREFIX + MODEL_SUFFIX + '.json').read())
    model.load_weights(ch.MODELDIR + ch.MODEL_WEIGHTS_PREFIX + MODEL_SUFFIX + '.h5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    ## PREDICT

    classes = model.predict(predict_data)

    ## REPRESENT
    
    sendmessage = ""

    for g in zip(geometry, classes[0]):
        sendmessage += ' '.join((g[0], ":", "{:.8f}%".format(g[1]) , "\n"))

    predict_index = np.argmax(classes)

    message =  "predicted:" + geometry[predict_index]
    sendmessage += "\n\n"
    sendmessage += "PREDICTED: " + geometry[predict_index]

    print message
    os.system("say " + message)

    return sendmessage


if __name__ == "__main__":

    host = '172.16.15.1'
    port = 9092

    listen_and_execute(host, port, sumsum)


