def imgread(path):
    img=cv2.imread(path)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def data_prepare(filename,correction):
    car_images=[]
    steering_angles=[]
    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            steering_center = float(row[3])
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            # read in images from center, left and right cameras
            path ='./data/' # fill in the path to your training IMG directory
            path0=path + row[0]
            path1=path + row[1]
            path2=path + row[2]
            img_center = cv2.imread(path0)  #process_image(np.asarray(cv2.imread(path + row[0])))
            #img_center=cv2.cvtColor(img_center,cv2.COLOR_BGR2RGB)
            img_left = cv2.imread(path1)  #process_image(np.asarray(cv2.imread(path + row[1])))
            #img_left=cv2.cvtColor(img_left,cv2.COLOR_BGR2RGB)
            img_right = cv2.imread(path2) #process_image(np.asarray(cv2.imread(path + row[2])))
            #img_right=cv2.cvtColor(img_right,cv2.COLOR_BGR2RGB)
            car_images.extend(img_center)
            #car_images.extend(img_left)
            #car_images.extend(img_right)
            steering_angles.append(steering_center)
            #steering_angles.append(steering_left)
            #steering_angles.append(steering_right)
    return car_images, steering_angles


def data_augment(images,measurements):
    augmented_images=[]
    augmented_measurements=[]
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement)
        augmented_measurements.append(measurement*-1.0)
    return augmented_images, augmented_measurements

import csv
import cv2
import numpy as np

csv_file='./data//driving_log.csv'
correction=0.25

images, measurements = data_prepare(csv_file, correction)

augmented_images, augmented_measurements=data_augment(images,measurements)

X_train=np.array(augmented_images)
y_train=np.array(augmented_measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Flatten())
#model.add(Dense(1))

model.add(Convolution2D(6, 5, 6, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 6, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')