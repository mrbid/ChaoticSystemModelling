# James William Fletcher (james@voxdsp.com)
#       C to Keras Bridge for Predictor
#               APRIL 2022
import sys
import os
import numpy as np
from tensorflow import keras
from os.path import isfile
from os.path import getsize
from os import remove
from struct import pack
from time import sleep

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

input_size = 96
model_name = sys.argv[1]

model = keras.models.load_model(model_name)
input_size_floats = input_size*4
while True:
        try:
                sleep(0.001)
                if isfile("/dev/shm/uc_input.dat") and getsize("/dev/shm/uc_input.dat") == input_size_floats:
                        with open("/dev/shm/uc_input.dat", 'rb') as f:
                                data = np.fromfile(f, dtype=np.float32)
                                remove("/dev/shm/uc_input.dat")
                                if data.size == input_size:
                                        input = np.reshape(data, [-1, input_size])
                                        r = model.predict(input)
                                        y = r.flatten()
                                        with open("/dev/shm/uc_r.dat", "wb") as f2:
                                                for x in y:
                                                        f2.write(pack('f', x))
        except Exception:
                pass
