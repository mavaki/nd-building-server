import tensorflow as tf
import numpy as np
import sys
from flask import Flask, request, jsonify

from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet')

# Start server
app = Flask(__name__)

# TODO: Register with catalog server (and spin up a thread to reregister at an interval

def classify_image(img_path):
    # load/preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # predict class probabilities
    predictions = base_model.predict(img_array)

    # decode top prediction
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    predicted_class = decoded_predictions[0][1] # class label
    return predicted_class

@app.route("/", methods=['GET'])
def home():
    return jsonify({'status': 'up'})

@app.route('/classify', methods=['POST'])
def classify_image():
    '''Given an image, generate a predicted label'''
    
    # Check if the request contains an image file 
    if 'image' not in request.files:
        print('image not in request')
        return jsonify({'error': 'No image found in request'}), 400
    
    # Save the image temporarily
    image = request.files['image']
    image_filename = f'tmp.jpg'
    image.save(image_filename)
    
    # Classify image
    predicted_class = classify_image(image_filename)
    print(f'predicted class: {predicted_class}')
    
    # Respond with predicted class
    return jsonify({'label': predicted_class}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
