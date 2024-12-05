import tensorflow as tf
import numpy as np
import sys, os
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

    # Remove image
    #os.remove('tmp.jpg')

    # Respond with predicted class
    return jsonify({'label': predicted_class}), 200

@app.route('/submit', methods=['POST'])
def submit_image():
    '''Save given image and label to directory'''

    # Check if the request contains an image file 
    if 'image' not in request.files or 'label' not in request.form:
        print('image and/or label not in request')
        return jsonify({'error': 'Request must contain image and label'}), 400

    image = request.files['image']
    label = request.form['label']

    # Validate that the label is provided
    if not label:
        return jsonify({'error': 'Label is required'}), 400

    # Save the image to a specific directory
    save_dir = '~/images'
    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.time()
    
    image_filename = os.path.join(save_dir, f'{image.filename}_{timestamp}.jpg')
    image.save(image_filename)

    # Optionally save the label in a text file (if needed)
    label_filename = os.path.splitext(image.filename)[0] + '_label.txt'
    label_filepath = os.path.join(save_dir, label_filename)

    with open(label_filepath, 'w') as label_file:
        label_file.write(label)

    # Respond with success message
    return jsonify({'message': 'Image and label saved successfully', 
                    'image_path': image_filename, 
                    'label_path': label_filepath}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
