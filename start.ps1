# start_services.ps1
# ----------------------------------------------------------------------------------
# NOTA: Para ejecutar este script en Windows, puede que necesites ajustar la política de ejecución de PowerShell.
# Abre una terminal de PowerShell como Administrador y ejecuta uno de los siguientes comandos:
#   Set-ExecutionPolicy RemoteSigned  (Recomendado para seguridad)
#   Set-ExecutionPolicy Bypass        (Menos seguro, pero útil para desarrollo local)
# Luego, puedes ejecutar el script con: .\start_services.ps1
# ----------------------------------------------------------------------------------

# Detiene el script si ocurre un error
$ErrorActionPreference = "Stop"

Write-Host "### Iniciando Servicios Necesarios para el Agente ###" -ForegroundColor Cyan

# --- Carga de variables de entorno desde .env ---
$envFilePath = ".\.env"
if (Test-Path $envFilePath) {
    # Lee el contenido, elimina los retornos de carro (CR) para compatibilidad, y procesa línea por línea
    Get-Content $envFilePath | ForEach-Object {
        $line = $_.Trim()
        # Ignorar líneas en blanco o comentarios
        if ($line -and !$line.StartsWith("#")) {
            # Dividir la línea solo en el primer '='
            $parts = $line.Split("=", 2)
            if ($parts.Length -eq 2) {
                $key = $parts[0].Trim()
                $value = $parts[1].Trim()
                # Establecer la variable de entorno para la sesión actual
                [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
                Write-Host ">> Variable de entorno cargada: $key" -ForegroundColor DarkGray
            }
        }
    }
} else {
    Write-Host ">> Archivo .env no encontrado" -ForegroundColor Red
    exit 1
}


# --- Creación de Entorno Virtual ---
if (-not (Test-Path ".\.venv")) {
    Write-Host ">> Creando entorno virtual..." -ForegroundColor Yellow
    # 'python' debe estar en el PATH del sistema
    python -m venv .\.venv
    
    Write-Host ">> Activando entorno e instalando dependencias..." -ForegroundColor Yellow
    # Activar el entorno virtual en PowerShell
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
} else {
    Write-Host ">> El entorno virtual ya existe. Activándolo." -ForegroundColor Green
    # Activar el entorno virtual si ya existe
    .\.venv\Scripts\Activate.ps1
}

Write-Host "### ✅ Todos los servicios están listos. ###" -ForegroundColor Cyan

# --- Ejecutar aplicación ---
Write-Host ">> Iniciando app.py..." -ForegroundColor Yellow
python app.py