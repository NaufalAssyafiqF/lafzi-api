import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder

# Load model and labels
model = tf.keras.models.load_model('src/app/model/model_alif_v2.h5')

my_label = ['false', 'true']

# Initialize Flask app
app = Flask(__name__)

# Function to process audio file and extract MFCC features
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
    return mfcc.T


@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Anda berhasil terhubung dengan API'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    # index_label = request.form.get('indexLabel')
    file = request.files['file']
    
    # if not index_label:
    #     return jsonify({"error": "No indexLabel part"})
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # try:
    #     # index_label_int = int(index_label)  # Convert indexLabel to integer
    # except ValueError:
    #     return jsonify({"error": "Invalid indexLabel value"})
    
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

        if index_akurasi > 0.7 and index_akurasi < 1.3 :
            isPredicted = True
        else :
            isPredicted = False
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({"predicted_label": predicted_label, 
                        'confidence': predictions.tolist(), 
                        'accuration_value': index_akurasi,
                        'is_predicted': isPredicted})

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # app.run(debug=True, port=5000)
    app.run(host='0.0.0.0', port=8080)
