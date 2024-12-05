import tensorflow as tf
import numpy as np
import socket
import sys

from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet')

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

def start_server(port):
    # set up the server socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('', port))
        server_socket.listen()
        
        # get the assigned port
        _, port = server_socket.getsockname()
        print(f'\nlistening on port {port}')
        
        while True:
            conn, addr = server_socket.accept()
            print(f'connected by {addr}')
            
            # receive the image size
            image_size = int(conn.recv(1024).decode())
            conn.sendall(b'SIZE RECEIVED')
            
            # receive the image data
            received_data = b''
            while len(received_data) < image_size:
                packet = conn.recv(4096)
                if not packet:
                    break
                received_data += packet

            # save the received image
            img_path = "images/image.jpg"
            with open(img_path, "wb") as f:
                f.write(received_data)
            
            print("image received and saved as 'image.jpg'")

            # classify the image
            predicted_class = classify_image(img_path)
            print(f'predicted class: {predicted_class}')
            
            # send classification result back to the client
            conn.sendall(predicted_class.encode())
            conn.close()

if __name__ == "__main__":
    # accept port from command line
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_server(port)
