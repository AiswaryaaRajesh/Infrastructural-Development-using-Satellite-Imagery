import os
import shutil
import sqlite3
import numpy as np
import cv2
import tensorflow as tf
import tflearn
import pandas as pd
from flask import Flask, send_from_directory, render_template, request, redirect, url_for
from random import shuffle
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__, static_folder="static")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

MODEL_NAME = 'infra-1e-3-2conv-basic.model'
IMG_SIZE = 50
LR = 1e-3
VERIFY_DIR = 'static/images'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/home')
def home():
    return render_template('./index.html')

   
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'filename' not in request.files:
            print("No file part in request")
            return redirect(url_for('home', msg='No file selected'))

        file = request.files['filename']
        if file.filename == '':
            print("File name is empty")
            return redirect(url_for('home', msg='No file selected'))

        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Preprocess image
        def process_verify_data():
            img_data = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_data is None:
                raise ValueError("Invalid image file.")
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            return np.array(img_data)

        img_data = process_verify_data()
        img_data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)

        # Build exact CNN used in cnn_vita.py
        tf.compat.v1.reset_default_graph()
        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')   # This is where it’s different
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 3, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists(f"{MODEL_NAME}.meta"):
            model.load(MODEL_NAME)
            print(f"✅ Model '{MODEL_NAME}' loaded successfully")
        else:
            print(f"❌ Model file '{MODEL_NAME}.meta' not found")
            return "Model file not found.", 500

        model_out = model.predict([img_data])[0]
        labels = {0: "Low Infrastructure", 1: "Medium Infrastructure", 2: "High Infrastructure"}
        label = labels[np.argmax(model_out)]
        accuracy = f"{model_out[np.argmax(model_out)] * 100:.2f}%"

        print(f"📊 Prediction: {label}, Accuracy: {accuracy}")

        return render_template(
            './result.html',
            label=label,
            accuracy_value=accuracy,
            ImageDisplay=f"/static/uploads/{file.filename}"
        )

    except Exception as e:
        print("Exception during image analysis:", str(e))
        return f"Internal server error: {str(e)}", 500



@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    data = request.form
    values = [float(data[key]) if key.replace('.', '', 1).isdigit() else data[key] for key in data]

    df = pd.read_csv("city_data.csv")
    X = df.drop(columns=["City"])
    y = df["Wealth Index"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = X.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(), categorical_features)
    ])

    clf = Pipeline([('preprocessor', preprocessor), ('classifier', LinearRegression())])
    clf.fit(X_train, y_train)
    new_data = pd.DataFrame([values[1:]], columns=X.columns)
    predicted_infra_level = clf.predict(new_data)[0]

    reasons = []
    if isinstance(values[1], (int, float)) and values[1] < 5:
        reasons.append('Low Infrastructure Development due to Wealth Index less than 5')
    if isinstance(values[2], (int, float)) and values[2] < 100:
        reasons.append('Low Infrastructure Development due to no. of Hospitals less than 100')
    if isinstance(values[3], (int, float)) and values[3] < 100:
        reasons.append('Low Infrastructure Development due to no. of Schools less than 100')
    if values[6] == 'Scarce':
        reasons.append('Low Infrastructure Development due to Scarce Water Resource')
    if isinstance(values[8], (int, float)) and values[8] > 1000:
        reasons.append('Low Infrastructure Development due to no. of Unemployment greater than 1000')

    str_label = ', '.join(reasons) if reasons else 'Good Infrastructure Development'

    return render_template('./result.html', status2=f'Predicted Infrastructure Development Level: {predicted_infra_level:.2f}', reasons=str_label)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
