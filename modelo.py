import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = 'data'
data_entrenar_dir = 'data/entrenar'
data_prueba_dir = 'data/prueba'

# Para entrenar el modelo vamos a usar una porcion de las imagenes
cant_img = len(list('data/*/*/*.jpg'))
print(cant_img)

# -- Model --
# Parametros de las imagenes utilizadas para entrenar
tam_batch = 32
alto_img = 224
ancho_img = 224

# Dividir el directorio de imagenes para usar una porcion para entrenar y otra para validacion
entrenar_ds = tf.keras.utils.image_dataset_from_directory(
  data_entrenar_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(alto_img, ancho_img),
  batch_size=tam_batch)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_entrenar_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(alto_img, ancho_img),
  batch_size=tam_batch)

# Clases de imagenes en el dataset, creadas por la libreria en funcion de las carpetas de imagenes 
clases = entrenar_ds.class_names
print(clases)

# -- Entrenar --

# Buffering de las imagenes para no bloquear el disco
AUTOTUNE = tf.data.AUTOTUNE

entrenar_ds = entrenar_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Reducir el rango de los colores de 0-255 a 0-1
normalization_layer = layers.Rescaling(1./255)
# TODO:  Traducir comentarios
# Apply layer to dataset
normalized_ds = entrenar_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# -- Crear el modelo --
num_clases = len(clases)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(alto_img, ancho_img, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_clases)
])

# -- Compilar el modelo --
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Display a summary
model.summary()

# -- Train the model --
epochs=10
history = model.fit(
  entrenar_ds,
  validation_data=val_ds,
  epochs=epochs
)

# -- Convertir y guardar el modelo para su uso posterior --

# Convert the model.
convertidor = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = convertidor.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)


# -- Display results --


# -- Test the model by telling it to identify new data --
benign_path = 'data/prueba/normal/1.jpg'

img = tf.keras.utils.load_img(
    benign_path, target_size=(alto_img, ancho_img)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predicciones = model.predict(img_array)
puntaje = tf.nn.softmax(predicciones[0])

print(
    "Esta imagen fue clasificada como lunar {} con {:.2f} porciento de confianza."
    .format(clases[np.argmax(puntaje)], 100 * np.max(puntaje))
)
