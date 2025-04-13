from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('fake_face_cnn_model.h5')

# Define the path for storing uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allow only specific file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if file is allowed
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load and preprocess the image
        img = Image.open(filename)
        img = img.resize((128, 128))  # Resize the image to match model input size
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict with the model
        prediction = model.predict(img_array)

        # Convert prediction into real/fake
        result = 'Fake' if prediction < 0.5 else 'Real'

        return jsonify({'prediction': result, 'filename': file.filename})

    return jsonify({'error': 'Invalid file format'})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
