#!/bin/bash

# Lista de librerias
dependencies=("tensorflow" "numpy" "matplotlib")

# Chequear si pip esta instalado
if ! command -v pip &> /dev/null; then
    echo "Pip no esta instalado. Por favor instalar pip y correr el script nuevamente. Para mas informacion leer el README.md en el repositorio de GitHub."
    exit 1
fi

# Iterar a traves de la lista de librerias
for dependency in "${dependencies[@]}"; do
    if ! pip show $dependency &> /dev/null; then
        echo "$dependency no esta instalado. Instalando..."
        pip install $dependency
    else
        echo "$dependency ya esta instalado."
    fi
done

echo "Chequeo de librerias e instalacion completado."