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

# Clases de imagenes, en nuestro caso son normales o cancerigenos
entrenar_ds = tf.keras.utils.image_dataset_from_directory(
  data_entrenar_dir,
  seed=123,
  image_size=(alto_img, ancho_img),
  batch_size=tam_batch)

# Clases de imagenes en el dataset, creadas por la libreria en funcion de las carpetas de imagenes 
class_names = entrenar_ds.class_names
print(class_names)

# -- Training --

# Buffering de las imagenes para no bloquear el disco
AUTOTUNE = tf.data.AUTOTUNE

entrenar_ds = entrenar_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Dont use the 255 values per color, use this instead
normalization_layer = layers.Rescaling(1./255)
# TODO: Cambiar todas las variables a espanol y chequear que funcione bien.
# Apply layer to dataset
normalized_ds = entrenar_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# -- Create the model --
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# -- Compile the model --
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Display a summary
model.summary()

# -- Train the model --
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# -- Display results --


# -- Test the model by telling it to identify new data --cc
benign_path = 'test/benign/1.jpg'

img = tf.keras.utils.load_img(
    benign_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
