from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
MODEL_PATH = 'model.h5'  # Update this if your model file is named differently
model = load_model(MODEL_PATH)

# List of class names (update this list according to your actual classes)
class_names = ['Zebra', 'Lion', 'Tiger', 'Elephant', 'Giraffe', 'Monkey', 'Bear', 'Panda', 'Kangaroo', 'Koala']

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have this HTML file for uploading

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request has a file part
    if 'file' not in request.files:
        return 'No file part. Please upload a file.'

    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return 'No selected file. Please upload a file.'

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Save the file in the 'static' folder
        file_path = os.path.join('static', filename)
        file.save(file_path)

        return redirect(url_for('predict', filename=filename))
    else:
        return 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'

@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join('static', filename)
    
    try:
        # Load and preprocess the image
        img = load_img(file_path, target_size=(32, 32))  # Resize image to 32x32 as per model's input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Normalize image pixel values (if your model expects normalized input)
        img_array = img_array / 255.0  # If required by your model
        
        # Predict the class
        predictions = model.predict(img_array)
        print(f"Raw Predictions: {predictions}")  # Print raw predictions to see the values
        
        predicted_class = np.argmax(predictions[0])  # Get the index of the highest probability
        result = class_names[predicted_class]  # Get the class name based on prediction
        
        # Calculate probabilities for all classes
        prob_dict = {class_names[i]: predictions[0][i] * 100 for i in range(len(class_names))}
        
        # Confidence level
        confidence_level = max(prob_dict.values())

        return render_template('result.html', 
                               filename=filename, 
                               result=result,
                               prob_dict=prob_dict, 
                               confidence_level=confidence_level)
    except Exception as e:
        return f'Error in prediction: {str(e)}'

if __name__ == '__main__':
    # Create the 'static' folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
