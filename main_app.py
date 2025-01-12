from flask import Flask, request, render_template
from flask import send_from_directory

import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

app = Flask(__name__)

menu = [{"name": "Курсовая", "url": "models"}]

classes = ['облачно', 'туманно', 'дождливо', 'солнечно', 'закат/восход']

model_dense = tf.keras.models.load_model('model/best_models/dense.keras')
model_convolutional = tf.keras.models.load_model('model/best_models/convolutional.keras')
model_convolutional_dense_mixed = tf.keras.models.load_model('model/best_models/convolutional_dense_mixed.keras')
model_transfer = tf.keras.models.load_model('model/best_models/transfer.keras')
model_detect = YOLO('detect/runs8/detect/yolov8-custom/weights/best.pt')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def preproces_image(image_path):
    img = image.load_img(image_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

import glob

@app.route('/', methods=['GET', 'POST'])
def kurs():
    if request.method == 'GET':
        return render_template('kurs.html', title="", menu=menu, class_model='', processed_image=None)

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        img_array = preproces_image(file_path)


        results = model_detect.predict(source=file_path, conf=0.35, save=True)


        output_dir = results[0].save_dir
        processed_image_path = glob.glob(f"{output_dir}/*.jpg")[0]


        if os.path.exists(processed_image_path):
            shutil.move(processed_image_path, os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename))
        else:
            return f"Processed image not found in {output_dir}"

        processed_image = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)

        model_method = np.array([str(request.form['method_param'])])

        if model_method == 'dense':
            predictions = model_dense.predict(img_array)
            prediction = np.argmax(predictions, axis=1)
        elif model_method == 'convolutional':
            predictions = model_convolutional.predict(img_array)
            prediction = np.argmax(predictions, axis=1)
        elif model_method == 'mixed':
            predictions = model_convolutional_dense_mixed.predict(img_array)
            prediction = np.argmax(predictions, axis=1)
        elif model_method == 'transfer':
            predictions = model_transfer.predict(img_array)
            prediction = np.argmax(predictions, axis=1)
        else:
            return 'wrong method'


    return render_template('kurs.html', title="Результаты", menu=menu,
                               class_model=f"Погода на фото:  {classes[prediction[0]]}",
                               processed_image=processed_image)

if __name__ == "__main__":
    app.run(debug=True)
