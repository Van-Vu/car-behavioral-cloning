import os
import csv
import cv2
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Activation, Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, Reshape
from keras import backend as K
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.regularizers import l2 as l2_reg

image_paths = []
measurements = []
adjusted_angle = 0.2
data_folder = './Clone/'
image_folder = data_folder + 'IMG/'

with open(data_folder + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        measurement = float(line[3])

        image_paths.append(line[0].split('\\')[-1])
        measurements.append(measurement)

        image_paths.append(line[1].split('\\')[-1])
        measurements.append(measurement + adjusted_angle)

        image_paths.append(line[2].split('\\')[-1])
        measurements.append(measurement - adjusted_angle)   
    
X_train = np.array(image_paths)
y_train = np.array(measurements)

## SPLIT TRAIN AND VALID DATA
X_train, y_train = shuffle(X_train, y_train)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2) 

def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
        image, steering = cv2.flip(image,1), -steering
    return image,steering

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = random.uniform(0.3,1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def crop_and_resize(image):
    crop_image = cv2.resize(image[60:140,:], (64,64))
    return crop_image

def preprocess_image(image_path,steering, isTrainImage = True):
    image = cv2.imread(image_folder + image_path)
    image = crop_and_resize(image)
    if isTrainImage:
        flip_image, flip_steering = random_flip(image, steering)
        flip_image = random_brightness(flip_image)
        return flip_image, flip_steering
    else:
        return image, steering

def getModel():
    model = Sequential()
    model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(64,64,3)))
    model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu",kernel_regularizer=l2_reg(0.0001)))
    model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu",kernel_regularizer=l2_reg(0.0001)))
    model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu",kernel_regularizer=l2_reg(0.0001)))
    model.add(Convolution2D(64,(3,3),activation="relu",kernel_regularizer=l2_reg(0.0001)))
    model.add(Convolution2D(64,(3,3),activation="relu",kernel_regularizer=l2_reg(0.0001)))
    model.add(Flatten())
    model.add(Dense(120,kernel_regularizer=l2_reg(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(50,kernel_regularizer=l2_reg(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(10,kernel_regularizer=l2_reg(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(1,kernel_regularizer=l2_reg(0.0001)))
    model.summary()
    return model

def generator(X_data, y_data, isTrainImages = True, batch_size=32):
    while 1: # Loop forever so the generator never terminates
        images, measurements = shuffle(X_data, y_data)
        
        batch_images = []
        batch_measurements = []
        for i in range(batch_size):
            choice = int(np.random.choice(len(images),1))
            image, measurement = preprocess_image(images[choice], measurements[choice], isTrainImages)
            
            batch_images.append(image)
            batch_measurements.append(measurement)
        
        yield np.array(batch_images), np.array(batch_measurements)
            
            
# compile and train the model using the generator function
train_generator = generator(X_train, y_train, True)
validation_generator = generator(X_valid, y_valid, False)

model = getModel()
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)
history_object = model.fit_generator(train_generator, steps_per_epoch= 3000, validation_data=validation_generator,
                    validation_steps=len(X_valid), epochs=20)

model.save('model.h5')
print('Model saved')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('trainingloss.jpg')

print('images saved')