from flask import Flask, request, jsonify,render_template
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
import cv2
import onnxruntime as rt
from PIL import Image


class_names=['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']


app = Flask(__name__)
@app.route('/',methods=['post','get'])
def homepage():
    return render_template('index.html')

model_path = 'vegetablemodelonnx.onnx'
@app.route('/predict', methods=['POST'])
def Prediction():
    img_path = request.files['image'].read()

    file_bytes = np.fromstring(img_path, np.uint8)
    img_path = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    start_time = time.time()
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    img = cv2.resize(img_path, (224, 224))      
    img_array = np.expand_dims(img, axis=0)
    image = np.array(img_array) / 255.0
    input_data = np.array(image, dtype=np.float32)
    output = sess.run([output_name], {input_name: input_data})

    end_time = time.time()
    output_data = output[0]
    prediction = np.argmax(output_data)
    prediction = class_names[prediction]
    inference_time = end_time - start_time

    return render_template('results.html', prediction=(prediction,'inference time: ',inference_time))


if __name__ == '__main__':
    app.run(debug=True)