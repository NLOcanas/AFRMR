C. AFRMR-ComputerVis3.py

#Programmer: Nathaniel L Ocanas
## HEC3-Senior Design- Advanced Mobile Fire Rescue Robot.
#Date 1-3-2021

## This code was developed using a template from Jay Bhatt on youtube for 2d ##Convolutional Neural Networks
## https://www.youtube.com/watch?v=uqomO_BZ44g
##There are also plots added to this code taken from ##Tensorflow.org/tutorials/images/classification. 

## The difference is this Neural Network is a categorical function because there are ## more than 2 classifications of images and using different
## image sizes.This NN is meant to classify Animals, Fires and Humans, The aim to is ## to get above 90% accuracy and a loss below 3%.
## This Neural Network was trained using GOOGLE-COLAB GPUs, Because this is meant to ## mound to a google drive the directories reflect
## these paths for the files. With this code different networks were trained with 
## different layers, learning rate, batch size, and activation functions.
##

##Mounting drive to GOOGLE-COLAB IDE
from google.colab import drive
drive.mount('/content/drive',force_remount=True)

## Connect to GOOGLE-COLAB GPU web-service
%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print("Num GPUs Available: ",len(tf.config.experimental.list_physical_devices('GPU')))

## Begin code for Training Data ( Fire, Human, Animal, Other)
## Calling all necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import datetime

## save date and time variable
ct = datetime.datetime.now()
strct=str(ct)
print("current time:-", strct)

# set up training and validation data
train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)

# Variable to terate through training dataset and create variable
train_dataset = train.flow_from_directory('/content/drive/MyDrive/Colab        Notebooks/basedata/train',target_size = (256,256),          # resize image in folder
            batch_size = (17),    #match batch size to amount of data being trained on class_mode ='categorical',    # use categorical class for training more than 2 classes
                              )   # Categorical because more than 2 outputs-non binary

# Variable to Iterate through validation dataset and create variable
validation_dataset = validation.flow_from_directory('/content/drive/MyDrive/Colab                   Notebooks/basedata/validation',target_size = (256,256),       # resize image in folder
    batch_size = (17),           # match batch size to amount of data being trained on
    class_mode ='categorical',# use categorical class for training more than 2 classes
                  )              # Categorical because more than 2 outputs-non binary


## Create regularizers to control over and under fitting models
regularizer = tf.keras.regularizers.L2(2.)
activity_regularizer=tf.keras.regularizers.L2(0.01)

## Define model layers : Here using sequential for 2D convolutional neural network 
## with relu activation functions in hidden layers and Softmax activation function at ## output

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),padding="same",activation ='relu', input_shape=(256,256,3)),                                      ## input layer
                                     
##1st hidden layer w 32 filters==>nodes
                                     tf.keras.layers.Conv2D(32,(3,3),padding="same",activation ='relu'), # increasing nodes to 32
              tf.keras.layers.MaxPool2D(2,2),

##2nd hidden layer w 64  filers==nodes                                     tf.keras.layers.Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(0.001),padding="same",activation ='relu'),                                       # increasing nodes to 64
              tf.keras.layers.MaxPool2D(2,2),
                                    
##3rd hidden layer
##tf.keras.layers.Conv2D(128,(3,3),kernel_regularizer=regularizers.l2(0.001),padding="same",activation ='relu'),                                   # increasing nodes to 128
##tf.keras.layers.MaxPool2D(2,2),
                                    
## 4th hidden layer w 256  filers==nodes                                   tf.keras.layers.Conv2D(256,(3,3),kernel_regularizer=regularizers.l2(0.001),padding="same",activation ='relu'),                                     # increasing nodes to 256
   tf.keras.layers.MaxPool2D(2,2),

## flatten data for Dense layer, this layer is 2D.
## tf.keras.layers.Flatten(),
tf.keras.layers.GlobalAveragePooling2D(),
                                    
##1st Dense Layer with regularizers and dropouts
tf.keras.layers.Dense(1024,activation ='relu',                                                    #kernel_initializer='zeros',                                                           activity_regularizer=regularizers.l2(0.001),                                                           kernel_regularizer=regularizers.l2(0.01)),
tf.keras.layers.Dropout(.23, input_shape=(224,224,3)),

##2nd Dense Layer with regularizers and dropouts
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(1024,activation ='relu',                                                        activity_regularizer=regularizers.l2(0.001),                                                      kernel_regularizer=regularizers.l2(0.01)),
tf.keras.layers.Dropout(.25, input_shape=(224,224,3)),
                              

## Output Layer 
##categorical needs 4 output nodes for 4 categories given
tf.keras.layers.Dense(4,activation ='softmax',)      # Softmax activation function for 
                                                     ## categorical-non-binary output
                                     ])


## Compile model with RMSprop function as optimizer and lr(learning rate)
model.compile(loss='categorical_crossentropy',optimizer = RMSprop(lr=0.0007),
                   ## Resource: Jay Bhatt- https://www.youtube.com/watch?v=uqomO_BZ44g
              metrics = ['accuracy'])

## Set Epoch number
epochs=1000 #epoch iterations===> 1000 ~ 95% accuracy

##fit model and define number of steps in epochs 
model_fit = model.fit(train_dataset,
                      steps_per_epoch= 4,                   # steps inside each epoch
                      epochs =epochs,
                      validation_data = validation_dataset,
                      )#callbacks=[callback])
## print model summary to terminal
model.summary()


##  Save trained model to folder with weights
model.save('/content/drive/MyDrive/Colab Notebooks/visionNN/V4AMFRR_vision1000epochs.b17pool2.2rms.1dense.wdo.25.lr.0007') 

# Save model to file
acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


## Plot Accuracy and Loss, Resource: TensorFlow.org/tutorials/images/classification
epochs_range = range(epochs)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.show()
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

## save plot as image in folder
#https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displa#ying-it-using-matplotlib

plt.savefig('/content/drive/MyDrive/Colab Notebooks/visionNN/Accuracy and Loss plots/V4AMFRR_vision1000epochs.b17pool2.2rms.1dense.wdo.25.lr.0007.png',bbox_inches='tight')

## test other data outside of training data to see if model is correctly classifying ## images.
##
##for i in os.listdir(dir_path):
##     img = image.load_img(dir_path+'//'+i,target_size=(300,300))
##     plt.imshow(img)
##     plt.show()
##
##     X = image.img_to_array(img)
##     X=np.expand_dims(X,axis = 0)
##     images = np.vstack([X])
##     val = model.predict(images)
##     val=val.astype(int)
##     newval = np.split(val[0],3)
##     print(newval)     
##    
##     for item in newval[0]:
##          if  item == 1:
##                   print("animal")
##          else:
##               print("")
##     for item in newval[1]:
##          if  item == 1:
##                   print("fire")
##          else:
##               print("")
##     for item in newval[2]:
##             if  item == 1:
##                   print("human")
##             else:
##                    print("")


     
##indice = train_dataset.class_indices
##print(indice)
####clss = train_dataset.classes
##print(clss)
