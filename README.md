# Clasificador de lunares cancerigenos

### Archivos
- **setupLinux.sh** Es el archivo encargado de chequear que las librerias necesarias esten instaladas e instalandolas si no lo estan para Linux.
- **setupWindows.ps1** Es el archivo encargado de chequear que las librerias necesarias esten instaladas e instalandolas si no lo estan para Windows.
- **entrenar.py** Es el archivo encargado de generar y entrenar el modelo para luego converirlo y guardarlo como modelo TensorFlow  Lite.
- **prediccion.py** Es el archivo encargado de correr el modelo guardado para realizar pruebas.

### Modo de uso
El programa se corre mediante el uso de uno consola de comandos, se requiere tener instalado Python y las librerias tensorflow y matlpotlib(Este ultimo punto se puede hacer corriendo el script setupLinux.sh/setupWindowsps1 dependiendo del sistema operativo.
#### **Entrenamiento**
Correr el programa modelo.py mediante el comando ```python modelo.py```, de esa manera se entrenara un modelo secuencial Keras a partir de las imagenes contenidas en el directorio entrenar/. Luego el modelo es guardado como TensorFlow Lite modelo(```model.tflite```).
![Alt text](entrenamiento.png?raw=true "Entrenamiento")
#### **Carga y ejecucion**
El modelo entrenado se puede correr utilizando el programa prediccion.py mediante el comando ```python prediccion.py```, el programa preguntara al usuario por la direccion de la imagen que se desea clasificar y luego devolvera los resultados.
![Alt text](prueba.png?raw=true "Prueba")
