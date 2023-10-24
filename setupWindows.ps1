# Lista de librerias
$dependencies = @("tensorflow", "numpy", "matplotlib")

# Chequear si pip esta instalado
if (-not (Get-Command -Name pip -ErrorAction SilentlyContinue)) {
    Write-Host "Pip no esta instalado. Por favor instalar pip y correr el script nuevamente. Para mas informacion leer el README.md en el repositorio de GitHub."
    exit 1
}

# Iterar a traves de la lista de librerias
foreach ($dependency in $dependencies) {
    if (-not (Get-Command -Name $dependency -ErrorAction SilentlyContinue)) {
        Write-Host "$dependency no esta instalado. Instalando..."
        pip install $dependency
    }
    else {
        Write-Host "$dependency ya esta instalado."
    }
}

Write-Host "Chequeo de librerias e instalacion completado."
