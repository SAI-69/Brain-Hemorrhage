from flask import Flask, request, render_template, jsonify,redirect,url_for
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping 
from skimage.feature import hog
import pickle
import io

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the CNN model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

def process_and_predict(file):
    # Load and preprocess the image
    image = Image.open(file)
    image = image.resize((224, 224))  # Adjust based on your model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = image_array.reshape(1, 224, 224, 3)  # Reshape for the model (batch size, height, width, channels)

    # Make prediction (logits/raw predictions)
    logits = model.predict(image_array)

    # Apply softmax to convert logits to probabilities
    probabilities = tensorflow.nn.softmax(logits).numpy()

    # Get the predicted class (class with the highest probability)
    predicted_class = np.argmax(probabilities, axis=1)

    # Return the predicted class and probabilities for both classes (hemorrhage and no hemorrhage)
    return predicted_class.tolist(), probabilities[0]  # probabilities[0] contains the softmax output for each class

# Define the route to handle file upload and display the results
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from the form submission
        file = request.files['file']
        filename = file.filename

        # Call process_and_predict to get the prediction and probabilities
        predicted_class, probabilities = process_and_predict(file)

        # Extract the probabilities for class 0 (no hemorrhage) and class 1 (hemorrhage)
        no_hemorrhage_prob = probabilities[0]
        hemorrhage_prob = probabilities[1]

        # Render the template with the prediction and probabilities
        return render_template('results.html',
                               prediction=predicted_class,
                               hemorrhage_prob=hemorrhage_prob,
                               no_hemorrhage_prob=no_hemorrhage_prob,
                               filename=filename)
if __name__ == '__main__':
    app.run(debug=True)
