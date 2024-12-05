import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import ast

# Download base model pre-trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet')

img_path = 'hatchet.jpg'
img = image.load_img(img_path)

img = img.resize((224, 224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) # Expand dimensions to fit model input (model expects a batch dimension)
img_array = img_array / 255.0  # scale pixel values to [0, 1]

prediction = base_model.predict(img_array)
predicted_class = np.argmax(prediction[0])
predicted_class

# get class name from class number

with open('imagenet1000.txt') as filein:
  labels = ast.literal_eval(filein.read())

print('class number:', predicted_class)
print(labels[predicted_class])


