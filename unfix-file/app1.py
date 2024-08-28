import os
import pickle
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

app = Flask(__name__)

# Konfigurasi path
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
MODEL_PATH = 'model-23juni-full-vfull2.h5'  # Path ke file model .h5 Anda
LABELS_PATH = 'class_indices-23juni2.pkl'  # Path ke file labels .pkl Anda

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan label
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, 'rb') as f:
    labels = pickle.load(f)

class_labels = {v: k for k, v in labels.items()}

# Fungsi untuk mengecek ekstensi file yang diizinkan
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk mengekstrak Mel Spectrogram
def extract_mel_spectrogram(audio_path, img_size=(224, 224)):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    
    fig, ax = plt.subplots()
    img = ax.imshow(S_dB, origin='lower', aspect='auto', cmap='viridis')  # Anda dapat mengganti 'viridis' dengan cmap lain
    fig.colorbar(img, ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel Frequency')
    fig.set_size_inches(img_size[0] / 100, img_size[1] / 100)
    ax.axis('off')

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Anda berhasil terhubung dengan API'})

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Ekstrak Mel Spectrogram
            mel_spectrogram = extract_mel_spectrogram(filepath)
            mel_spectrogram = mel_spectrogram / 255.0  # Normalisasi
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

            # Prediksi
            prediction = model.predict(mel_spectrogram)
            predicted_label_index = np.argmax(prediction)
            predicted_label = class_labels[predicted_label_index]

            response = jsonify({'prediction': predicted_label})
        
        finally:
            # Hapus file setelah prediksi selesai
            os.remove(filepath)
        
        return response
    
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)




# Pastikan gambar memiliki ukuran (224, 224)
    # fig = plt.figure(figsize=(img_size[0] / 100, img_size[1] / 100))
    # librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    # plt.axis('off')
    # Visualisasi Mel Spectrogram menggunakan Matplotlib

# def extract_mel_spectrogram(audio_path, img_size=(224, 224)):
#     y, sr = librosa.load(audio_path, sr=None)
#     S = librosa.feature.melspectrogram(y=y, sr=sr)
#     S_dB = librosa.power_to_db(S, ref=np.max)

#     fig, ax = plt.subplots()
#     img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
#     fig.set_size_inches(img_size[0] / 100, img_size[1] / 100)
#     ax.axis('off')

#     # Convert to numpy array
#     fig.canvas.draw()
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#     plt.close(fig)
#     return data