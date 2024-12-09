import os
import time
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the building classifier model
model = load_model("building_classifier_model.h5")

# Define a Flask app
app = Flask(__name__)

# Load class indices from the test generator
def get_class_indices():
    test_dir = '/home/ec2-user/training_images'  # Adjust this to your training directory
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator.class_indices

class_indices = get_class_indices()
class_labels = {v: k for k, v in class_indices.items()}


def predict_image(model, img_path):
    """Predict the class of the given image using the loaded model."""
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence_scores = predictions[0]
        predicted_confidence = float(confidence_scores[predicted_class_idx] * 100)  # Convert to percentage and ensure float

        return class_labels.get(predicted_class_idx, "Unknown"), predicted_confidence

    except Exception as e:
        return str(e), None

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'server is running'})

@app.route('/classify', methods=['POST'])
def classify():
    """Endpoint to classify a building image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Save the image temporarily
    image = request.files['image']
    image_filename = f"tmp_{int(time.time())}.jpg"
    image.save(image_filename)

    # Predict class
    try:
        predicted_label, predicted_confidence = predict_image(model, image_filename)
        print('prediction:', predicted_label)
        return jsonify({
            'label': predicted_label,
            'confidence_percent': predicted_confidence
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up
        os.remove(image_filename)

@app.route('/submit', methods=['POST'])
def submit_image():
    """Save given image and label to directory."""
    if 'image' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Request must contain image and label'}), 400

    image = request.files['image']
    label = request.form['label']

    # Validate label
    if not label:
        return jsonify({'error': 'Label is required'}), 400

    save_dir = '/home/ec2-user/images'
    os.makedirs(save_dir, exist_ok=True)

    timestamp = str(time.time()).replace('.', '')
    name = f"{os.path.splitext(image.filename)[0]}_{timestamp}"
    image_filename = os.path.join(save_dir, f"{name}.jpg")
    image.save(image_filename)

    label_filepath = os.path.join(save_dir, f"{name}.txt")
    with open(label_filepath, 'w') as label_file:
        label_file.write(label)

    return jsonify({
        'message': 'Image and label saved successfully',
        'image_path': image_filename,
        'label_path': label_filepath
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

