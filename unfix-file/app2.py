from flask import Flask, request, jsonify
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# from keras.models import load_model
import pickle
import io
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'model-21juli-full-vfull2.h5'
# CLASS_INDICES_PATH = 'class_indices-21juli.pkl'
model = tf.keras.models.load_model(MODEL_PATH)
my_label = ['1 alif', '10 ro', '11 za', '12 sin', '13 syin', '14 shod', '15 dhod', '16 tho', '17 zho', '18 ain', '19 ghoin', '2 ba', '20 fa', '21 qof', '22 kaf', '23 lam', '24 mim', '25 nun', '26 wau', '27 haa', '28 ya', '3 ta', '4 tsa', '5 jim', '6 hha', '7 kho', '8 dal', '9 dzal']

# # Load class indices
# with open(CLASS_INDICES_PATH, 'rb') as f:
#     class_indices = pickle.load(f)

# Reverse the class indices to get class labels
# class_labels = {v: k for k, v in class_indices.items()}

UPLOAD_FOLDER = 'upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



def extract_mfcc_from_audio(audio_bytes):
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=10)
    mfcc = mfcc_transform(waveform)
    mfcc = mfcc.numpy().squeeze()
    mfcc = (mfcc - mfcc.mean()) / mfcc.std()  # Normalisasi
    return mfcc



def resize_mfcc_to_rgb(mfcc,filename):
    # Normalize MFCC to [0, 1]
    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
    # Apply the 'viridis' colormap
    cm = plt.get_cmap('viridis')
    mfcc_colored = cm(mfcc)[:, :, :3]  # Ignore the alpha channel
    # Resize to (224, 224)
    mfcc_image = Image.fromarray((mfcc_colored * 255).astype(np.uint8))
    mfcc_image = mfcc_image.resize((224, 224))
    mfcc_rgb = np.array(mfcc_image)

    mfcc_image.save(filename)

    return mfcc_rgb

@app.route('/', methods=['GET'])
def index():
    with open('class_indices-21juli.pkl', 'rb') as f:
        label_names = pickle.load(f)
    return jsonify({'message': 'Anda berhasil terhubung dengan API', 'labels': label_names})



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    audio_bytes = file.read()
    mfcc = extract_mfcc_from_audio(audio_bytes)

    # Save MFCC image
    mfcc_filename = os.path.join(UPLOAD_FOLDER, 'mfcc_image.png')
    mfcc_rgb = resize_mfcc_to_rgb(mfcc, mfcc_filename)

    # Assuming the model expects a 4D tensor with shape (1, 224, 224, 3)
    mfcc_rgb = np.expand_dims(mfcc_rgb, axis=0)  # Add batch dimension

    # Use the model to predict
    prediction = model.predict(mfcc_rgb)
    predicted_class_index = np.argmax(prediction)
    predicted_class = my_label[predicted_class_index]

    return jsonify({
        'prediction': predicted_class,
        'confidence': prediction.tolist(),
    })

if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     audio_bytes = file.read()
#     mfcc = extract_mfcc_from_audio(audio_bytes)

#     # Resize MFCC to RGB image
#     mfcc_rgb = resize_mfcc_to_rgb(mfcc)

#     # Assuming the model expects a 4D tensor with shape (1, 224, 224, 3)
#     mfcc_rgb = np.expand_dims(mfcc_rgb, axis=0)  # Add batch dimension

#     # Use the model to predict
#     prediction = model.predict(mfcc_rgb)
#     predicted_class_index = np.argmax(prediction)
#     predicted_class = class_labels[predicted_class_index]

#     return jsonify({
#         'prediction': predicted_class,
#         'confidence': prediction.tolist()
#     })


# def resize_mfcc_to_rgb1(mfcc, filename):
#     mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())  # Scale to [0, 1]
#     mfcc = (mfcc * 255).astype(np.uint8)  # Scale to [0, 255]
#     mfcc_image = Image.fromarray(mfcc)
#     mfcc_image = mfcc_image.resize((224, 224)).convert('RGB')  # Resize and convert to RGB
#     mfcc_rgb = np.array(mfcc_image)


#     mfcc_image.save(filename)
#     return mfcc_rgb