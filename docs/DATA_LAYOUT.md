# Estructura de datos esperada

Los datos crudos reales no se incluyen en este paquete. Sí se incluye un
XDF sintético en `data/synthetic/` para probar los scripts sin exponer
datos privados. Para reproducir el flujo con datos propios, usar una
estructura local equivalente:

```text
data/
  raw/
    P01_run01_full.xdf
    P02_run01_visual_audio2.xdf
  logs/
    P01_run01_trials.csv
    P01_run01_blocks.csv
    P01_run01_session_log.csv
  derived/
    visual_alpha_features.csv
    auditory_erp_features.csv
  reports/
    figures/
```

## Streams XDF esperados

- EEG: stream tipo señal, usualmente 250 Hz, con etiquetas de canales.
- Marcadores: stream de eventos de tarea.
- Eye tracker/gaze: opcional, si se adquirió.
- Audio de referencia: opcional, útil para auditoría temporal.

Los scripts permiten indicar nombres de streams por argumento cuando los
nombres locales difieren.

El XDF sintético incluido es deliberadamente mínimo: contiene EEG simulado y
marcadores de tarea. No incluye `BCI_Audio`, `EyeFix_Gaze`,
`EyeFix_Markers` ni `BCI_SyncProbe`. Esos streams se documentan como módulos
opcionales de adquisición, no como requisito del análisis offline.
