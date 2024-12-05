from flask import Flask, request, jsonify
import os
import time

app = Flask(__name__)

# Create the directories to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set the folder for uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET'])
def home():
    return jsonify({"message": "up"})

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains files
    if 'image' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'No image or label found in request'}), 400
    
    # Get the image and label
    image = request.files['image']
    label = request.form['label']
    
    timestamp = time.time()

    # Save the image with a unique name
    image_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'image_{timestamp}.jpg')
    image.save(image_filename)
    
    # Save the label to a text file
    label_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'label_{timestamp}.txt')
    with open(label_filename, 'w') as f:
        f.write(label)
    
    # Respond with a success message
    return jsonify({'message': 'Image and label uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

