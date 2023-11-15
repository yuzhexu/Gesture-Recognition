from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
# Create a TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path="ASL_Model/model.tflite")
# Allocate memory for the model's input and output tensors
interpreter.allocate_tensors()
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assume input_data is preprocessed data
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Extract output data
output_data = interpreter.get_tensor(output_details[0]['index'])
