import streamlit as st
import numpy as np


st.title('Image Classification')
st.write("Cat vs Dog Classification")
#Testing the models
import tensorflow
import keras
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img 

# creating a object
model = keras.models.load_model('cat_dog.keras')
path=st.file_uploader("Upload an image")
if st.button('Submit'):
#    class_labels={0:'cat',1:'dog'}
# path = '/content/flower/val/rose/24841052213_90fc2b1046_c.jpg'
#path = '/content/drive/MyDrive/new_set_belt/new_seat_belt_REFINED_RAW_DATASET/test/positive/opencv_frame_28 (4).png'
   test_image = load_img(path,target_size = (64,64,3)) #224 224
   test_image = image.img_to_array(test_image)
   test_image = np.expand_dims(test_image,axis = 0)
   result = model.predict(test_image)
   print(result)
#    result = np.argmax(result)
   output = [result][0][0]
   if output <0.5:
       output="Cat"
   else:
       output="Dog"
   st.write(f"Predicted class: {output}")
   st.image(path)
# # reading the image
# testImage = img.imread(path)

# # displaying the modified image
# plt.imshow(testImage)