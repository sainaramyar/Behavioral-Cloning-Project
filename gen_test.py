import csv
import cv2
import numpy as np
import random
correction=0.25
lines=[]

from sklearn.utils import shuffle

def my_resize(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (64,64))

with open('./data//driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

nrows=len(lines)

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for j in range(3):
                    source_path = batch_sample[j].strip()#.split('/')[-1]
                    filename = source_path.split('/')[-1]
                    local_path = 'data/IMG/' + filename
                    image = cv2.imread(local_path)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    images.append(image)
                correction = 0.25
                measurement = float(batch_sample[3])
                measurements.append(measurement)
                measurements.append(measurement + correction)
                measurements.append(measurement - correction)

            augmented_images = []
            augmented_measurements = []
        # augment data with flipped version of images and angles
            for image, measurement in zip(images, measurements):
                if abs(measurement)>0.8:
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image,1))
                    augmented_measurements.append(measurement*-1.0)
                else:
                    if random.uniform(0, 1)>0.7:
                        augmented_images.append(image)
                        augmented_measurements.append(measurement)
                        augmented_images.append(cv2.flip(image,1))
                        augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            #yield shuffle(X_train, y_train)
            yield (X_train, y_train)


num_training_lines = int(0.8*nrows)
batch_size=32
BATCH_SIZE=32
training_lines = lines[0:num_training_lines-1]
validation_lines = lines[num_training_lines:]

train_generator = generator(training_lines,batch_size)
validation_generator = generator(validation_lines, batch_size)

for i in range(3):
    x_batch, y_batch = next(train_generator)
    print(x_batch.shape, y_batch.shape)