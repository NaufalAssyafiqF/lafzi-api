import os
import numpy as np
import librosa
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)

# Load model and labels
model = tf.keras.models.load_model('model/makharijul_huruf_model_038.h5')
# label_encoder = None
# with open('label_encoder.npy', 'rb') as f:
#     label_encoder = np.load(f, allow_pickle=True).item()
my_label = ['1 alif', '10 ro', '11 za', '12 sin', '13 syin', '14 shod', '15 dhod', '16 tho', '17 zho', '18 ain', '19 ghoin', '2 ba', '20 fa', '21 qof', '22 kaf', '23 lam', '24 mim', '25 nun', '26 wau', '27 haa', '28 ya', '3 ta', '4 tsa', '5 jim', '6 hha', '7 kho', '8 dal', '9 dzal']

# Fungsi untuk mengekstrak mel-spectrogram
def extract_mel_spectrogram(file_path, n_mels=128, max_len=32):
    y, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)  # Normalisasi
    
    if mel_db.shape[1] > max_len:
        mel_db = mel_db[:, :max_len]
    else:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    return mel_db

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Process the audio file
        mel_spectrogram = extract_mel_spectrogram(file_path)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension
        # mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(mel_spectrogram)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = my_label[predicted_index]

        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({"predicted_label": predicted_label, "confidence": predictions.tolist()})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
