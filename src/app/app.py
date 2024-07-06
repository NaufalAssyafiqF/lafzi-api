from flask import Flask, request, jsonify
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array 
import numpy as np
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from PIL import Image

app = Flask(__name__)

import os
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

# Load model and class indices
MODEL_PATH = 'model-23juni-full-vfull2.h5'
CLASS_INDICES_PATH = 'class_indices-23juni2.pkl'
model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, 'rb') as f:
    class_indices = pickle.load(f)

class_labels = {v: k for k, v in class_indices.items()}

def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')
    mel_path = 'mel_spectrogram.png'
    plt.savefig(mel_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return mel_path

def prepare_image(image_path, target_size):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Anda berhasil terhubung dengan API'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)

    # Extract mel-spectrogram
    mel_spectrogram_path = extract_mel_spectrogram(audio_path)

    # Prepare image
    image = prepare_image(mel_spectrogram_path, target_size=(224, 224))

    # Predict
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]
    predicted_probability = float(np.max(predictions[0]))

    # Clean up
    os.remove(audio_path)
    os.remove(mel_spectrogram_path)

    return jsonify({
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'predicted_probability': predicted_probability
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
