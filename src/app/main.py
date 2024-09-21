import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Labels for prediction
my_label = ['false', 'true']

# Initialize Flask app
app = Flask(__name__)

# Function to process audio file and extract MFCC features
# def extract_mfcc(file_path, n_mfcc=13):
#     y, sr = librosa.load(file_path, sr=None)

#     # Noise reduction using simple spectral subtraction
#     S = np.abs(librosa.stft(y))
#     noise_pow = np.mean(S[:, :100], axis=1, keepdims=True)  # Estimate noise power
#     S_dn = np.maximum(S - noise_pow, 0)
#     y_clean = librosa.istft(S_dn)

#     # Extract MFCC features
#     mfcc = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=n_mfcc)
#     mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
#     return mfcc.T

def reduce_noise(y, sr):
    noise_sample = y[0:int(0.25 * sr)]  # Assume first 0.25 seconds is noise
    noise_profile = np.mean(librosa.feature.mfcc(y=noise_sample, sr=sr, n_mfcc=13), axis=1)
    
    # Apply noise reduction by subtracting the noise profile from the original signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_cleaned = mfcc - noise_profile[:, np.newaxis]
    
    return librosa.feature.inverse.mfcc_to_audio(mfcc_cleaned, sr=sr)

# Function to process audio file and extract MFCC features after noise reduction
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    
    # Reduce noise in the audio
    y_clean = reduce_noise(y, sr)
    
    # Extract MFCC from the cleaned audio
    mfcc = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
    return mfcc.T

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Anda berhasil terhubung dengan API'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    if 'huruf' not in request.form:
        return jsonify({"error": "No huruf part"})

    try:
        huruf_index = int(request.form['huruf'])
    except ValueError:
        return jsonify({"error": "Invalid huruf value. Must be an integer."})

    # Ensure huruf is within the correct range
    if huruf_index < 1 or huruf_index > 28:
        return jsonify({"error": "Invalid huruf value. Must be between 1 and 28."})

    # Load the correct model based on the "huruf" index
    model_path = f'model/model_{huruf_index}.h5'
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model for huruf {huruf_index} not found."})

    model = tf.keras.models.load_model(model_path)

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Process the audio file
        mfcc = extract_mfcc(file_path)
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(mfcc)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = my_label[predicted_index]

        index_akurasi = predictions.tolist()[0][1]

        if index_akurasi > 0.7 and index_akurasi < 1.3:
            isPredicted = True
        else:
            isPredicted = False

        # Clean up uploaded file
        os.remove(file_path)

        return jsonify({
            "predicted_label": predicted_label,
            'confidence': predictions.tolist(),
            'accuration_value': index_akurasi,
            'is_predicted': isPredicted
        })

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
