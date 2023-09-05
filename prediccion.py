import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tam_batch = 32
alto_img = 224
ancho_img = 224

TF_MODEL_FILE_PATH = 'model.tflite' # Directorio donde se guerdo el TensorFlow Lite model

interpretador = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

# Imprimir en consola los parametros de entrada y de salida del modelo
print(interpretador.get_signature_list())

# TODO: Comentar que hacer las lineas.
classify_lite = interpretador.get_signature_runner('serving_default')
classify_lite

benign_path = 'data/prueba/normal/1.jpg'

img = tf.keras.utils.load_img(
    benign_path, target_size=(alto_img, ancho_img)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)



