import os
import helpers

def shellquote(s):
    return "'" + s.replace("'", "'\\''") + "'"

def resizeImage(from_image, to_image):
    command = "convert " + shellquote(from_image) + " -resize " + str(helpers.IMAGE_WIDTH) + "x" + str(helpers.IMAGE_HEIGHT) + " " + shellquote(to_image)
    os.system(command) # I KNOW THIS IS HORRIBLE

for path, subdirs, files in os.walk(helpers.RAW_TRAINING_DIR):
    for name in files:
        fullpath = os.path.join(path, name)
        newfullpath = fullpath.replace(helpers.RAW_SUFFIX, helpers.REDUCED_SUFFIX)

        print ">> SHOULD CONVERT", fullpath, "TO",  newfullpath

        if os.path.isfile(newfullpath):
            print "BUT FILE EXISTS"
        else:
            resizeImage(fullpath, newfullpath)



