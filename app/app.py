from flask import Flask, render_template, redirect, url_for, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from PIL import Image

app = Flask(__name__,  static_url_path='/static')

model = tf.keras.models.load_model("saved_model(1).h5")

app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

@app.route("/")
def homepage():
    return render_template('index.html')

@app.route("/panduan")
def panduan():
    return render_template('panduan.html')

@app.route("/aplikasi")
def aplikasi():
    return render_template('aplikasi.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Preprocess the uploaded image
    image = request.files['file']
    filename = secure_filename(image.filename)
    image.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    predicted_image = '/static/images/uploads/' + filename
    img = Image.open(image)
    img_resize = img.resize((256, 256))
    img_array = np.array(img_resize)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values between 0 and 1

    # Make predictions on the input image
    predictions = model.predict(img_array)
    
    # Return the predicted class label

    return render_template('result.html', predictions=predictions, predicted_image=predicted_image)

if __name__ == '__main__':
    app.run()

