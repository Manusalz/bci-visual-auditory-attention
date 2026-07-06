# Standalone Eye Fixation Monitor

Monitor opcional para control ocular grueso con webcam. Emite muestras LSL
continuas (`EyeFix_Gaze`) y marcadores LSL (`EyeFix_Markers`) para registrar
desviaciones grandes de fijacion durante una tarea.

Este modulo no es necesario para correr el pipeline offline del paper. Se
incluye como ejemplo publico de adquisicion auxiliar. Fue probado en Windows
con Python 3.11; Linux y macOS no fueron validados.

## Instalacion

```powershell
cd acquisition\eye_fixation_monitor
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uso rapido

Preview con webcam:

```powershell
.\.venv\Scripts\python.exe main.py --mode preview
```

Prueba sin webcam, util para verificar LSL y archivos de salida:

```powershell
.\.venv\Scripts\python.exe main.py --mode headless --demo-mode simulated --duration-s 10
```

Autocalibracion guiada:

```powershell
.\.venv\Scripts\python.exe main.py --mode preview --autocalibrate
```

## Streams LSL

- `EyeFix_Gaze`: muestras continuas con gaze proxy, confianza, estado de
  fixbreak y variables de geometria/cabeza.
- `EyeFix_Markers`: eventos `eye/fixbreak/start`, `eye/fixbreak/end` y
  `eye/manual/test`.

## Configuracion

El archivo `config.yaml` es una plantilla publica. No contiene calibraciones de
participantes. En una sesion real se deben revisar:

- `camera.index`
- `geometry.eye_to_screen_cm`
- `geometry.screen_width_cm`
- `threshold.degree_radius`
- `threshold.invert_x`
- `threshold.gain_x` y `threshold.gain_y`

Si `auto_calibration.persist_to_yaml` queda en `true`, el programa puede
guardar parametros calibrados en `config.yaml`. Antes de publicar resultados,
no subir configuraciones con valores especificos de una persona.

## Privacidad

No subir a GitHub videos, capturas de webcam, carpetas `outputs/`, archivos de
calibracion individual ni logs de sesiones reales.
