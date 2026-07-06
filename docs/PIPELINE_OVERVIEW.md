# Visión general del pipeline

1. **Adquisición:** EEG y marcadores se guardan en XDF mediante LSL.
2. **Inspección XDF:** se listan streams, canales, muestras, duración y
   frecuencia efectiva.
3. **Inspección multimodal:** se grafica EEG con marcas de tarea, fases,
   targets y respuestas.
4. **Control ocular:** se cruzan marcadores de fijación, video local y
   ventanas de análisis para excluir épocas afectadas.
5. **QC EEG:** se revisan amplitudes, transitorios, saturación y ruido
   repetitivo.
6. **Features visuales:** potencia alfa posterior canónica 8-12 Hz y
   análisis exploratorio con frecuencia alfa individual.
7. **Features auditivas:** ERP/P300 centro-parietal ante targets atendidos,
   targets ignorados y estándares.
8. **Clasificación:** validación cruzada estratificada, BA, bootstrap y
   permutación.
9. **Reporte:** tabla auditada por contraste, figuras y decisiones de QC.
