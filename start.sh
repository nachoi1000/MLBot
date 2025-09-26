#!/bin/bash
# start_services.sh
echo "### Iniciando Servicios Necesarios para el Agente ###"

#!/bin/bash
set -e

# Convertir .env de formato Windows (CRLF) a Unix (LF) automáticamente
if [ -f .env ]; then
  # Usar sed para eliminar retornos de carro \r (caracter ^M)
  sed -i 's/\r$//' .env
else
  echo ">> Archivo .env no encontrado"
  exit 1
fi

# Cargar variables de entorno desde .env
set -a  # exportar automáticamente
source .env
set +a

# Crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo ">> Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo ">> Installing dependencies..."
    pip install -r requirements.txt
else
    echo ">> Virtual environment already exists. Skipping requirements installation."
    source .venv/bin/activate
fi

echo "### ✅ Todos los servicios están listos. ###"

# Ejecutar aplicación
echo ">> Starting app.py..."
python app.py