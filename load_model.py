You've used 86% of your storage. … If you run out of storage, you won't be able to upload new files.Learn more
def comp_vision():
     """ predict input image"""
          
     from tensorflow import keras
     from tensorflow.keras.preprocessing import image
     import tensorflow as tf
     import cv2
     import os
     import matplotlib.pyplot as plt
     import numpy as np

     model = keras.models.load_model('AMFRR_vision')
     #uimage='basedata/test/'+filename
     uimage= input('Enter the full name of the image in the test folder: ')

     img = image.load_img(uimage, target_size=(500,200))
     plt.imshow(img)
     plt.show()

     X = image.img_to_array(img)
     X=np.expand_dims(X,axis = 0)
     images = np.vstack([X])
     val = model.predict(images)
     val=val.astype(int)
     newval = np.split(val[0],3)
     print(newval)    

     for item in newval[0]:
          if  item == 1:
                print("animal")
          else:     
               print("")
     for item in newval[1]:
          if  item == 1:
                print("fire")
          else:
               print("")
     for item in newval[2]:
               if  item == 1:
                    print("human")
               else:
                    print("")
