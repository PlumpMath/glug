## Grasshopper Voxel Classification experiment

![160612_prototype_1.gif](https://github.com/dantaeyoung/glug/blob/master/voxel_classify/MEDIA/160612_prototype_1.gif)

#### Description:

---> [**Click for Vimeo description/explanation.**](https://www.youtube.com/watch?v=pn8yuOofb4U) <---

#### Requirements

A computer running (real) Python, a computer running Rhino/GH/IronPython.

I used OSX and an instance of Windows/Rhino/GH/IronPython running in VMWare. This may be possible all on the same computer in Windows between 'normal Python' and IronPython, but I haven't tried it.

[GangGang](https://github.com/dantaeyoung/GangGang/) is used to send data between both instances of Python. Some other Python libraries and GH libraries may be necessary.

#### Instructions (may require some tweaking):

1. `Classify_1_generate_training_set.ghx`. This should generate randomly rotated tetrahedra, cylinders, cones, and rectangular prisms, rotated, voxelized, and output as a 1-D string of 0/1 to indicate voxel or not-voxel.
2. `Classify_2_train_model.py`. Trains a model in Keras, saves it.
3. `Classify_5_predict_GangGangServer.py`. Loads Keras model, opens a GangGang server, waits for messages from client (Grasshopper).
4. `Classify_3_voxelize_and_send.ghx`. In GH, grabs the selected object, voxelizes it as JSON data, sends it via GangGang to server, gets result back. As a nice bonus, even says the result out loud on OSX. 

#### Loose notes/thoughts:

- IronPython is a pain. Some method to create a 'headless' Python server and to run real Python code in Grasshopper would be amazing. Also, really looking forward to [GH_CPython](https://github.com/MahmoudAbdelRahman/GH_CPython).
- There's probably a more simple way to do this with regression.
- Alternatives to voxelization (e.g. exploding a Brep into its components, grabbing area, silhouette length, orientation) may be interesting.
- I thought a 3D or even a 2D convolutional net would work better, but the best approach I found via trial-and-error was to have three simple fully-connected middle layers. My entirely unfounded guess: The voxelization is done around a square bounding box, which given the random rotation of most objects, leaves a lot of blank space in the eight corners of the box. I'm assuming that those eight corners are the most significant / easy ways to predict the geometry, and so through training, the fully connected layers eventually begin to best isolate those features. Since the geometry's pretty simple, there aren't repeating features that a convolutional net would be able to isolate; maybe 3D convolutional nets will better work on clusters of objects.
- A more deliberate use-case would better hone the role of classification.
- A better and meaning-rich spatial/3d object data set would really help in creating more interesting applications. AKA, ImageNet for 3d object data.
- This process of "creating your own training data, training a model, and using it" may have some more fun applications - process-wise, a hybrid of interactive genetic algorithms and ML. See: [Quadcopter Navigation in the Forest using Deep Neural Networks](https://youtu.be/umRdt3zGgpU?t=1m40s)

