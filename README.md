# Pipeline reproducible BCI EEG visual/auditivo

Este repositorio acompaña un estudio piloto de interfaz cerebro-computadora
con EEG de baja densidad para comparar atención visual encubierta y atención
auditiva selectiva dentro de cada participante.

El objetivo del paquete es documentar un flujo reproducible de adquisición
XDF, inspección multimodal, control ocular, control de calidad y análisis
offline. No incluye datos crudos privados.

Los scripts fueron preparados y probados en Windows con Python 3.11. El
pipeline offline debería ser portable, pero Linux/macOS no fueron validados en
este paquete.

## Qué incluye

- Scripts genéricos para inspeccionar XDF y visualizar ventanas de EEG con
  marcas de tarea.
- Un XDF sintético con EEG simulado, marcadores visuales/auditivos y
  respuestas con espacio de alto rendimiento.
- Taxonomía reproducible de eventos auditivos.
- Clasificador tabular con validación cruzada, exactitud balanceada,
  permutaciones y bootstrap.
- `paper_reproducibility/` con features anonimizadas por época, scripts para
  recalcular BA/IC/p, tabla de auditoría final y figuras generadas por código.
- Módulos opcionales de adquisición para publicar `BCI_Audio`,
  `EyeFix_Gaze` y `EyeFix_Markers` por LSL, junto con un selector demo de
  experimentos.
- Pruebas sintéticas mínimas para chequear que la taxonomía y el clasificador
  funcionan.

## Qué no incluye

- XDF crudos.
- Videos de cámara o eye tracker.
- Capturas de pantalla con rostro.
- Logs con nombres, fechas reales o rutas locales.
- Identificadores internos de adquisición.
- El runner privado completo del laboratorio.
- XDF crudos, timestamps absolutos y logs originales necesarios para una
  reproducción desde señal cruda. La reproducción pública del manuscrito parte
  de features derivadas anonimizadas.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux/macOS:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Los módulos opcionales de adquisición tienen dependencias separadas:

```powershell
.\.venv\Scripts\python.exe -m pip install -r acquisition\requirements_acquisition.txt
```

## Chequeo rápido

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\inspect_xdf.py scripts\auditory_event_roles.py scripts\classify_eeg_features.py scripts\plot_xdf_eeg_markers.py scripts\plot_xdf_window.py
.\.venv\Scripts\python.exe tests\test_event_taxonomy.py
.\.venv\Scripts\python.exe tests\test_classify_eeg_features.py
.\.venv\Scripts\python.exe tests\test_synthetic_xdf_loads.py
```

## Flujo reproducible

1. Adquirir EEG y marcas con LabRecorder/LSL en formato XDF.
2. Inspeccionar streams:

```powershell
python scripts\inspect_xdf.py --xdf data\raw\P01_run01_full.xdf --out outputs\P01_streams.csv
```

También puede probarse con el XDF sintético incluido:

```powershell
python scripts\inspect_xdf.py --xdf data\synthetic\P01_synthetic_high_performance.xdf --out outputs\synthetic_streams.csv
```

3. Revisar ventanas de EEG, marcas y fases:

```powershell
python scripts\plot_xdf_eeg_markers.py --xdf data\raw\P01_run01_full.xdf --start 0 --duration 120 --channels 6,7,8 --save outputs\P01_visual_window.png
```

Para el ejemplo sintético:

```powershell
python scripts\plot_xdf_eeg_markers.py --xdf data\synthetic\P01_synthetic_high_performance.xdf --start 0 --duration 90 --channels 6,7,8 --save outputs\synthetic_visual_window.png
```

El visor fue probado con el XDF sintético incluido y con un XDF real local del
laboratorio. Los datos reales no se distribuyen en este repositorio.

4. Construir features offline de alfa visual o ERP/P300 auditivo con el
pipeline local del laboratorio o con scripts equivalentes que exporten una
tabla CSV.

5. Clasificar features tabulares:

```powershell
python scripts\classify_eeg_features.py --features outputs\features.csv --label condition --positive target --feature-cols PO7_alpha_db,PO8_alpha_db,PZ_alpha_db --out outputs\classification_summary.json --permutations 1000 --bootstrap 2000
```

6. Reportar para cada contraste: archivo/corrida desidentificada, clases,
número de épocas por clase, hits, misses, falsas alarmas, tipo de análisis,
ventana o banda, balanced accuracy, intervalo bootstrap y p de permutación.

## Documentación

- `docs/ACQUISITION_MODULES.md`: módulos opcionales para `BCI_Audio`, control
  ocular y selector demo.
- `docs/EVENT_TAXONOMY.md`: clases auditivas y diferencias entre estándar,
  target atendido, target con respuesta y target ignorado.
- `docs/DATA_LAYOUT.md`: estructura esperada de datos locales no incluidos.
- `docs/QC_RULES.md`: reglas de control de calidad y exclusión.
- `docs/VALIDATION_STRATEGY.md`: clasificación, validación y azar empírico.
- `docs/EXPLORATORY_P300_XDAWN.md`: analisis complementario opcional con
  xDAWN/OAS/Tangent desde epocas derivadas locales no publicadas.
- `docs/GITHUB_UPLOAD_CHECKLIST.md`: pasos recomendados antes de publicar.
- `paper_reproducibility/`: reproducción de los resultados principales del
  manuscrito desde features derivadas anonimizadas.

## Licencia

Este repositorio se distribuye bajo licencia MIT, definida en `LICENSE`.
