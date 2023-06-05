import onnx
import tensorflow as tf
import tf2onnx
print(tf.__version__)
model = tf.keras.models.load_model('vegetablemodel.h5')
print('sdknf')

onnx_model, _ = tf2onnx.convert.from_keras(model)

# Save the ONNX model to a file
onnx_filename = 'vegetablemodelonnx.onnx'
onnx.save(onnx_model, onnx_filename)
