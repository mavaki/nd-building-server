import os
import time
import numpy as np
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import threading, subprocess, socket, json
import io
import zipfile

def register_with_catalog(name, port):
    # Recall function every 60 seconds
    t = threading.Timer(60.0, register_with_catalog, args=[name, port]).start()

    # Catalog entry
    entry = { "type" : "magic-polaroid",
             "owner" : "mvankir2",
             "port" : port,
             "project" : name
           }

    # Set addr and define UDP socket
    addr = ('catalog.cse.nd.edu', 9097)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    # Send UDP packet to name server
    sock.connect(addr)
    sock.sendall(json.dumps(entry).encode())
    sock.close()

# Get ip and send to catalog server
ip = subprocess.run(['dig', 'ANY','+short', '@resolver2.opendns.com', 'myip.opendns.com'], capture_output=True, text=True).stdout.strip()
register_with_catalog(ip, '8080')

# Load the building classifier model
model = load_model("building_classifier_model.h5")

# Define a Flask app
app = Flask(__name__)

# Load class indices from the test generator
def get_class_indices():
    test_dir = '/home/ec2-user/training_images'
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

# Generate images folder and subfolders to store new training data
IMAGE_FOLDER = '/home/ec2-user/training_images'
for class_label in class_labels.values():
    subfolder = os.path.join(IMAGE_FOLDER, class_label)
    os.makedirs(subfolder, exist_ok=True)

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
    image_filename = f"tmp_{time.time_ns()}.jpg"
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

def zip_folder(folder_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)  # Preserve folder structure
                zip_file.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer

@app.route('/submit', methods=['POST'])
def submit():
    """Save given image and label to directory for future model training."""
    if 'image' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Request must contain image and label'}), 400

    image = request.files['image']
    label = request.form['label']

    # Validate label
    if not label:
        return jsonify({'error': 'Label is required'}), 400
    if label not in class_names.values():
        return jsonify({'error': 'Class does not exist'}), 400

    # Save image in appropriate class folder
    save_dir = os.path.join(IMAGE_FOLDER, label)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = str(time.time()).replace('.', '')
    image_filename = os.path.join(save_dir, f"new_{timestamp}.jpg")
    image.save(image_filename)

    return jsonify({
        'message': 'Image added to training data'
    }), 200

@app.route('/training-data-check', methods=['GET'])
def training_data_check():
    if not os.path.exists(IMAGE_FOLDER):
        return jsonify({"new-image-count": 0})

    # Count how many images have been received 
    image_count = 0
    for class_label in class_labels.values():
        subfolder = os.path.join(IMAGE_FOLDER, class_label)
        image_count += sum(os.path.isfile(os.path.join(subfolder, f)) and f.startswith("new") for f in os.listdir(subfolder)) 

    return jsonify({'new-image-count': image_count}), 200

@app.route('/training-data', methods=['GET'])
def training_data():
    if not os.path.exists(IMAGE_FOLDER):
        return jsonify({"error": 0})

    # Zip the training data
    zip_data = zip_folder(IMAGE_FOLDER)
    return send_file(
        zip_data,
        mimetype='application/zip',
    ), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
