from flask import Flask, render_template, request

from tensorflow.keras.models import load_model   # Ensure TensorFlow is imported correctly

# Rest of your code remains the same...

import cv2
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model('D:\A\image style transfer\my_local_model.keras\my_local_model.keras')  # Update with the correct path

def preprocess_image(image):
    # Preprocess the image (resize, convert to grayscale, normalize)
    # Return the processed image
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    processed_image = cv2.resize(processed_image, (200, 200))  # Resize to model's input shape
    processed_image = processed_image.astype('float32') / 255.0  # Normalize
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=-1)  # Add channel dimension
    return processed_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file uploaded')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', prediction='No file selected')
        
        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            # Assuming your classes are labeled A-Z (0-25)
            predicted_letter = chr(predicted_class + ord('A'))
            return render_template('index.html', prediction=f'Predicted Letter: {predicted_letter}')

if __name__ == '__main__':
    app.run(debug=True)
