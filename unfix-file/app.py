import matplotlib
matplotlib.use('Agg')  # Set backend matplotlib menjadi non-GUI
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array 
import numpy as np
import pickle
import torchaudio
import torchaudio.transforms as transforms
import torch
from PIL import Image
import librosa
import tensorflow as tf
import threading
import os


app = Flask(__name__)

# Load model and class indices
MODEL_PATH = 'model-10juli-full-vfull2.h5'
CLASS_INDICES_PATH = 'class_indices-10juli.pkl'
model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, 'rb') as f:
    class_indices = pickle.load(f)

class_labels = {v: k for k, v in class_indices.items()}

# Parameter Mel Spectrogram (samakan dengan kode pelatihan)
sample_rate = 22050
n_fft = 4096
hop_length = 256
n_mels = 256
img_width = 224
img_height = 224


# Fungsi ekstraksi Mel Spectrogram (disesuaikan)
def extract_mel_spectrogram(audio_path, img_size=(img_width, img_height), output_path='mel_spectrogram.png'):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    mel_spectrogram = transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )(waveform)

    # Konversi ke numpy
    mel_spectrogram = mel_spectrogram.numpy().squeeze()

    # Power to DB (samakan dengan kode pelatihan)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Resize (sesuaikan jika perlu)
    plt.figure(figsize=(img_width / 100, img_height / 100))  
    plt.imshow(mel_spectrogram, origin='lower', aspect='auto', cmap='viridis')  
    plt.axis('off')

    # Simpan
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return output_path


def prepare_image(image_path, target_size):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Fungsi prediksi yang akan dijalankan di thread terpisah
def predict_audio(audio_path):
    mel_spectrogram_path = extract_mel_spectrogram(audio_path)
    image = prepare_image(mel_spectrogram_path, target_size=(img_width, img_height))
    
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]
    predicted_probability = float(np.max(predictions[0]))

    # Hapus file audio dan mel spectrogram sementara
    os.remove(audio_path)
    os.remove(mel_spectrogram_path)
    
    return {
        'predicted_class': int(predicted_class),
        'predicted_label': predicted_label,
        'predicted_probability': predicted_probability,
        'mel_spectrogram_path': mel_spectrogram_path
    }


@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Anda berhasil terhubung dengan API'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    # Gunakan threading untuk menjalankan prediksi di thread terpisah
    def prediction_thread():
        result = predict_audio(audio_path)
        return jsonify(result)  # Kirim hasil prediksi dalam bentuk JSON

    thread = threading.Thread(target=prediction_thread)
    thread.start()
    return jsonify({'message': 'Prediction in progress. Results will be sent when ready.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



# def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512):
#     # Load audio file using Torchaudio
#     waveform, sample_rate = torchaudio.load(audio_path)
#     waveform = waveform.squeeze(0)  # Ensure mono audio

#     # Create MelSpectrogram transform
#     mel_spectrogram_transform = transforms.MelSpectrogram(
#         sample_rate=sample_rate,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         n_mels=n_mels
#     )

#     # Apply transform
#     mel_spectrogram = mel_spectrogram_transform(waveform)
#     mel_spectrogram = 20 * torch.log10(mel_spectrogram + 1e-6)  # Convert to decibels

#     # Normalize (Optional)
#     mean = mel_spectrogram.mean()
#     std = mel_spectrogram.std()
#     mel_spectrogram_normalized = (mel_spectrogram - mean) / std
    
#     # Convert to image format (PIL)
#     mel_spectrogram_np = mel_spectrogram_normalized.squeeze().numpy()  
#     mel_spectrogram_np = (mel_spectrogram_np - np.min(mel_spectrogram_np)) / (np.max(mel_spectrogram_np) - np.min(mel_spectrogram_np))
#     mel_spectrogram_image = Image.fromarray((mel_spectrogram_np * 255).astype(np.uint8))

#     # Save as PNG
#     mel_path = 'mel_spectrogram.png'
#     plt.imsave(mel_path, mel_spectrogram_np, cmap='viridis', origin='lower')
    
#     return mel_path

# dg librosa
# def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512):
#     y, sr = librosa.load(audio_path, sr=None)
#     S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
#     S_DB = librosa.power_to_db(S, ref=np.max)
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
#     plt.axis('off')
#     mel_path = 'mel_spectrogram.png'
#     plt.savefig(mel_path, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     return mel_path

# def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512):
#     # Load audio file using librosa
#     y, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate

#     # Compute mel spectrogram using librosa
#     mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

#     # Convert to log scale (dB)
#     mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

#     # Save as PNG image using matplotlib
#     fig, ax = plt.subplots()
#     img = librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
#     fig.colorbar(img, ax=ax, format='%+2.0f dB')
#     ax.set_axis_off()
#     mel_path = 'mel_spectrogram.png'
#     plt.savefig(mel_path, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)

# dg tensorflow
# def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512):
#     # Load audio file using TensorFlow
#     audio_binary = tf.io.read_file(audio_path)
#     waveform, sample_rate = tf.audio.decode_wav(audio_binary)
#     waveform = tf.squeeze(waveform, axis=-1)

#     # Compute Short-Time Fourier Transform (STFT)
#     stft = tf.signal.stft(waveform, frame_length=n_fft, frame_step=hop_length, fft_length=n_fft)
#     spectrogram = tf.abs(stft)

#     # Compute mel spectrogram
#     num_spectrogram_bins = stft.shape[-1]
#     linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
#         num_mel_bins=n_mels,
#         num_spectrogram_bins=num_spectrogram_bins,
#         sample_rate=tf.cast(sample_rate, tf.float32),
#         lower_edge_hertz=0,
#         upper_edge_hertz=tf.cast(sample_rate, tf.float32) / 2)
#     mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
#     mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

#     # Convert to log scale
#     log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

#     # Normalize and convert to image (optional)
#     mel_spectrogram_normalized = (log_mel_spectrogram - tf.reduce_min(log_mel_spectrogram)) / (tf.reduce_max(log_mel_spectrogram) - tf.reduce_min(log_mel_spectrogram))
#     mel_spectrogram_image = tf.cast(mel_spectrogram_normalized * 255, tf.uint8)

#     # Save as PNG image
#     mel_path = 'mel_spectrogram.png'
#     plt.imsave(mel_path, mel_spectrogram_image.numpy(), cmap='viridis', origin='lower')

#     return mel_path