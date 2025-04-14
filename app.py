from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = load_model('fake_face_cnn_model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        img = Image.open(filename)
        img = img.resize((128, 128))  
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  
        prediction = model.predict(img_array)

        result = 'Fake' if prediction < 0.5 else 'Real'

        return jsonify({'prediction': result, 'filename': file.filename})

    return jsonify({'error': 'Invalid file format'})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
