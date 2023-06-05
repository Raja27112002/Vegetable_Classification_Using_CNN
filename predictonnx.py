from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import onnxruntime as rt
from PIL import Image

class_names=['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

''' This function is resposible for transforming the input given by user and predicting the output '''

class Prediction():
    def __init__(self):
        pass

    def Prediction(model_path,img_path):
        file = request.files['image'].read()
        sess = rt.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        image = Image.open(img_path)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        input_data = np.expand_dims(image, axis=0)
        input_data = np.array(input_data, dtype=np.float32)
        output = sess.run([output_name], {input_name: input_data})
        output_data = output[0]
        prediction = np.argmax(output_data)
        prediction = class_names[prediction]
        return prediction


model_path = 'vegetablemodelonnx.onnx'

