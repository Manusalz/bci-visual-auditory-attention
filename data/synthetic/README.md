# XDF sintético

`P01_synthetic_high_performance.xdf` es un archivo artificial generado
para probar el flujo público. No contiene EEG real, audio real, video,
nombre, fecha de adquisición ni dato biométrico de participantes.

El ejemplo simula:

- EEG de 8 canales a 250 Hz con nombres `FZ`, `CZ`, `P3`, `PZ`, `P4`,
  `PO7`, `PO8`, `OZ`;
- stream de marcadores `BCI_Markers`;
- bloque visual con cues izquierda/derecha, targets y respuestas;
- bloque auditivo `audio2` con estándares, targets atendidos, targets
  ignorados y respuestas con alto rendimiento.

`P01_synthetic_high_performance_events.csv` contiene la tabla de eventos
usada para construir el XDF.
