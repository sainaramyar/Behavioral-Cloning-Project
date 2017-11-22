import csv
import cv2
import numpy as np
import random
correction=0.25
lines=[]

def my_resize(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (64,64))

with open('./data//driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
images=[]
measurements=[]
for line in lines:
    source_path0=line[0]
    source_path1=line[1]
    source_path2=line[2]
    filename0=source_path0.split('/')[-1]
    filename1=source_path1.split('/')[-1]
    filename2=source_path2.split('/')[-1]
    #current_path = '../data/IMG/'+filename
    path0='data/IMG/'+filename0
    path1='data/IMG/'+ filename1
    path2='data/IMG/'+ filename2
    image_center=cv2.imread(path0)
    image_center=cv2.cvtColor(image_center,cv2.COLOR_BGR2RGB)
    image_left=cv2.imread(path1)
    image_left=cv2.cvtColor(image_left,cv2.COLOR_BGR2RGB)
    image_right=cv2.imread(path2)
    image_right=cv2.cvtColor(image_right,cv2.COLOR_BGR2RGB)
    images.extend((image_center,image_left,image_right))
    measurement_center=float(line[3])
    measurement_left=measurement_center+correction
    measurement_right=measurement_center-correction
    measurements.extend((measurement_center,measurement_left,measurement_right))

augmented_images=[]
augmented_measurements=[]
for image, measurement in zip(images, measurements):
    if abs(measurement)>0.7:
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
    else:
        if random.uniform(0, 1)>0.6:
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            augmented_images.append(cv2.flip(image,1))
            augmented_measurements.append(measurement*-1.0)

X_train=np.array(augmented_images)
y_train=np.array(augmented_measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import tensorflow as tf

model=Sequential()

model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (1, 1))))

#model.add(Cropping2D(cropping=((70,25),(1,1))))
model.add(Lambda(my_resize)) 


model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')