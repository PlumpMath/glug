import os
import time
import helpers

procrastinating = False

if procrastinating:
    file_dir = helpers.RAW_TRAINING_DIR + "YES/"
else:
    file_dir = helpers.RAW_TRAINING_DIR + "NO/"

while(True):
    time.sleep(30)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = file_dir + "screencap-" + timestr + ".png"
    os.system("screencapture -x " + filename)

    print "captured",
    print "PROCRASTINATIN" if procrastinating else "WORKIN",
    print ":::", filename

