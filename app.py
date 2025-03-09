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

# Recommendations based on emotions
recommendations = {
    "Anger": {
        "movies": [
            "Joker (2019) – Todd Phillips",
            "Fight Club (1999) – David Fincher",
            "The Dark Knight (2008) – Christopher Nolan",
            "John Wick (2014) – Chad Stahelski",
            "Gladiator (2000) – Ridley Scott",
            "Scarface (1983) – Brian De Palma",
            "V for Vendetta (2005) – James McTeigue",
            "Kill Bill: Volume 1 (2003) – Quentin Tarantino",
            "Mad Max: Fury Road (2015) – George Miller",
            "The Revenant (2015) – Alejandro G. Iñárritu"
        ],
        "songs": [
            "Break Stuff – Limp Bizkit",
            "Killing in the Name – Rage Against the Machine",
            "Smells Like Teen Spirit – Nirvana (Kurt Cobain)",
            "Chop Suey! – System of a Down (Serj Tankian)",
            "Bodies – Drowning Pool (Dave Williams)",
            "Duality – Slipknot (Corey Taylor)"
        ]
    },
    "Fear": {
        "movies": [
            "The Conjuring (2013) – James Wan",
            "Hereditary (2018) – Ari Aster",
            "The Shining (1980) – Stanley Kubrick",
            "A Quiet Place (2018) – John Krasinski",
            "It (Chapter One) (2017) – Andy Muschietti",
            "Sinister (2012) – Scott Derrickson",
            "The Ring (2002) – Gore Verbinski",
            "Get Out (2017) – Jordan Peele",
            "Insidious (2010) – James Wan",
            "Paranormal Activity (2007) – Oren Peli"
        ],
        "songs": [
            "Thriller – Michael Jackson",
            "Disturbia – Rihanna",
            "Enter Sandman – Metallica (James Hetfield)",
            "Somebody’s Watching Me – Rockwell (feat. Michael Jackson)",
            "Creep – Radiohead (Thom Yorke)",
            "Bring Me to Life – Evanescence (Amy Lee)",
            "Highway to Hell – AC/DC (Bon Scott)",
            "Bury a Friend – Billie Eilish",
            "Psycho Killer – Talking Heads (David Byrne)",
            "Helter Skelter – The Beatles (Paul McCartney)"
        ]
    },
    "Happy": {
        "movies": [
            "The Pursuit of Happyness (2006) – Gabriele Muccino",
            "Forrest Gump (1994) – Robert Zemeckis",
            "Zindagi Na Milegi Dobara (2011) – Zoya Akhtar",
            "Inside Out (2015) – Pete Docter",
            "3 Idiots (2009) – Rajkumar Hirani",
            "La La Land (2016) – Damien Chazelle",
            "The Intern (2015) – Nancy Meyers",
            "Paddington 2 (2017) – Paul King",
            "Mamma Mia! (2008) – Phyllida Lloyd",
            "The Secret Life of Walter Mitty (2013) – Ben Stiller"
        ],
        "songs": [
            "Happy – Pharrell Williams",
            "Can't Stop the Feeling! – Justin Timberlake",
            "Uptown Funk – Mark Ronson (feat. Bruno Mars)",
            "Dancing Queen – ABBA",
            "On Top of the World – Imagine Dragons",
            "Don't Stop Me Now – Queen (Freddie Mercury)",
            "I Gotta Feeling – The Black Eyed Peas",
            "Best Day of My Life – American Authors",
            "Shake It Off – Taylor Swift",
            "Walking on Sunshine – Katrina and the Waves"
        ]
    },
    "Neutral": {
        "movies": [
            "Cast Away (2000) – Robert Zemeckis",
            "Lost in Translation (2003) – Sofia Coppola",
            "The Grand Budapest Hotel (2014) – Wes Anderson",
            "The Secret Life of Walter Mitty (2013) – Ben Stiller",
            "Forrest Gump (1994) – Robert Zemeckis",
            "The Terminal (2004) – Steven Spielberg",
            "The Social Network (2010) – David Fincher",
            "Chef (2014) – Jon Favreau",
            "A Beautiful Mind (2001) – Ron Howard",
            "Nomadland (2020) – Chloé Zhao"
        ],
        "songs": [
            "Counting Stars – OneRepublic (Ryan Tedder)",
            "Let Her Go – Passenger (Mike Rosenberg)",
            "The Scientist – Coldplay (Chris Martin)",
            "Somewhere Only We Know – Keane (Tom Chaplin)",
            "Boulevard of Broken Dreams – Green Day (Billie Joe Armstrong)",
            "Yellow – Coldplay (Chris Martin)",
            "Fade Into You – Mazzy Star (Hope Sandoval)",
            "Take It Easy – Eagles (Glenn Frey)",
            "Riptide – Vance Joy",
            "Budapest – George Ezra"
        ]
    },
    "Sad": {
        "movies": [
            "The Pursuit of Happyness (2006) – Gabriele Muccino",
            "Hachi: A Dog's Tale (2009) – Lasse Hallström",
            "Schindler’s List (1993) – Steven Spielberg",
            "Titanic (1997) – James Cameron",
            "A Star Is Born (2018) – Bradley Cooper",
            "Requiem for a Dream (2000) – Darren Aronofsky",
            "Me Before You (2016) – Thea Sharrock",
            "Grave of the Fireflies (1988) – Isao Takahata",
            "Manchester by the Sea (2016) – Kenneth Lonergan",
            "Blue Valentine (2010) – Derek Cianfrance"
        ],
        "songs": [
            "Someone Like You – Adele",
            "Fix You – Coldplay (Chris Martin)",
            "Yesterday – The Beatles (Paul McCartney)",
            "Tears in Heaven – Eric Clapton",
            "Everybody Hurts – R.E.M. (Michael Stipe)",
            "Say Something – A Great Big World & Christina Aguilera",
            "Let Her Go – Passenger (Mike Rosenberg)",
            "Hurt – Johnny Cash (cover of Nine Inch Nails)",
            "I Will Always Love You – Whitney Houston",
            "The Night We Met – Lord Huron"
        ]
    }
}

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

    return jsonify({
        "emotion": predicted_emotion,
        "recommendations": recommendations[predicted_emotion]
    })

if __name__ == '__main__':
    app.run(debug=True)