#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Model for diffrentiating dog and cat..


# In[2]:


#Importing the important modules.

from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import numpy as np


# Setting the dimensions of our image.

img_width,img_height=150,150

train_data_dir='data/train'
validation_data_dir='data/validate'
nb_train_samples=4000
nb_validation_samples=300
epochs=20
batch_size=100

if K.image_data_format() == 'channels_first' :
    input_shape=(3,img_width,img_height)
else :
        input_shape=(img_width,img_height,3)
        
train_datagen = ImageDataGenerator(
 rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(1./255)

#Configuration used for Testing

train_generator=train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_width,img_height),
batch_size=batch_size,
class_mode='binary')

validation_generator=test_datagen.flow_from_directory(
validation_data_dir,
target_size=(img_width,img_height),
batch_size=batch_size,
class_mode='binary')


# In[3]:


#Creating the Network

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.summary()


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.summary()


#Compiling the model

model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])


# In[4]:


#Viewing the first image in the training dataset

import matplotlib.pyplot as plt

first_image = plt.imread("data/train/cats/cat.1.jpg")
plt.imshow(first_image)
plt.show()


# In[5]:


#Configuration Used for Training

model.fit_generator(
train_generator,
steps_per_epoch=nb_train_samples // batch_size,
epochs=epochs,
validation_data=validation_generator,
validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')




# In[6]:


#Evaluating the accuracy
accuracy=model.evaluate(train_generator)


# In[7]:


#Printing the accuracy
accuracy


# In[8]:


img_pred = image.load_img('catblur.jpg',target_size = (150,150))
image_resized=plt.imshow(img_pred)
img_pred = image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred,axis=0)


# In[9]:


#Predicting the Model

result=model.predict(img_pred)
#ynew = model.predict_classes(img_pred)
ynew = model.predict_proba(img_pred)

if result[0][0]==1:
    print('Can be a dog', '--Probablity', ynew )
   # prediction=" Is a Dog"
else :
    print('Can be a cat', '--Probablity', ynew)
   # prediction=" Is a Cat"
    


# In[ ]:




