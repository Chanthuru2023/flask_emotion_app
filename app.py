from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load the saved models
generator_path = r'C:\Users\USER\Downloads\generator_model_new-20240723T164058Z-001\generator_model_new'
discriminator_path = r'C:\Users\USER\Downloads\discriminator_model_new-20240723T164100Z-001\discriminator_model_new'

generator = tf.saved_model.load(generator_path)
discriminator = tf.saved_model.load(discriminator_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']
        file_bytes = file.read()

        # Preprocess the image
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))  # Resize to match the input shape of the generator
        img = (img - 127.5) / 127.5  # Normalize the image to [-1, 1]
        img = np.expand_dims(img, axis=0)

        # Generate the image using GAN generator
        noise = tf.random.normal([1, 100])
        generated_image = generator.signatures["serving_default"](tf.constant(noise))['conv2d_transpose_7'].numpy()
        generated_image = (generated_image[0] * 127.5 + 127.5).astype(np.uint8)

        # Convert the generated image to base64
        _, buffer = cv2.imencode('.png', generated_image)
        generated_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'generated_image': generated_image_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
