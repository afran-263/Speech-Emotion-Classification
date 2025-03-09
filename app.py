import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Define emotion labels
emotions = {"Anger": 0, "Fear": 1, "Happy": 2, "Neutral": 3, "Sad": 4}
n_mfcc = 40  # Number of MFCC features
DATASET_PATH = "data"
MODEL_PATH = "speech_emotion_model.h5"

def extract_features_from_audio(audio_data):
    try:
        audio, sample_rate = librosa.load(BytesIO(audio_data), sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Check if model exists, else train it
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    print("Model not found. Training new model...")

    # Load dataset and extract features
    X, y = [], []
    for speaker in os.listdir(DATASET_PATH):
        speaker_path = os.path.join(DATASET_PATH, speaker)
        if os.path.isdir(speaker_path):
            for emotion in os.listdir(speaker_path):
                emotion_path = os.path.join(speaker_path, emotion)
                if os.path.isdir(emotion_path) and emotion in emotions:
                    for file in os.listdir(emotion_path):
                        if file.endswith(".wav"):
                            file_path = os.path.join(emotion_path, file)
                            features = extract_features_from_audio(open(file_path, "rb").read())
                            if features is not None:
                                X.append(features)
                                y.append(emotions[emotion])

    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes=len(emotions))

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Reshape for CNN input
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Model definition
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(n_mfcc, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(emotions), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train model
    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

@app.route('/')
def home():
    return render_template('index.html')

# Define Flask API route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    audio_data = file.read()  # Read file without saving

    features = extract_features_from_audio(audio_data)
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    features = features.reshape(1, -1, 1)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    predicted_emotion = list(emotions.keys())[list(emotions.values()).index(predicted_label)]

    return jsonify({"emotion": predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)
