#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

keras = tf.keras
model = tf.keras.models.load_model('fazou.h5')

# Load the image
img = Image.open('test.jpg')

# Preprocess the image
img = img.resize((160, 160))  # Resize the image to match the model input size
img = np.array(img) / 255.0  # Normalize the pixel values to be between 0 and 1

# Make a prediction using your model
prediction = model.predict(np.array([img]))

if prediction > 0 : 
    print("Dog")
else :
    print("Cat")
