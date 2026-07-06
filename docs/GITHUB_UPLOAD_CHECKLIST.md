# Checklist antes de subir a GitHub

## Subir

- Todo el contenido de esta carpeta pública.
- Scripts genéricos en `scripts/`.
- Documentación en `docs/`.
- Configuraciones de ejemplo en `config/`.
- Tests sintéticos en `tests/`.

## No subir

- XDF crudos.
- Videos de webcam o eye tracker.
- Capturas de pantalla con rostro.
- Logs originales con nombres, fechas reales o rutas locales.
- El repositorio privado completo de adquisición/análisis.
- Documentos de trabajo con nombres propios o comentarios no publicables.
- Tablas exploratorias, resultados intermedios no congelados o material
  suplementario con identificadores privados. La carpeta `paper_reproducibility/`
  contiene únicamente derivados anonimizados seleccionados para el manuscrito.

## Antes de publicar

1. Elegir licencia. Recomendación práctica: MIT para código y CC-BY-4.0
   para documentación, si todos los autores están de acuerdo.
2. Definir cita del repositorio. Idealmente crear un release en GitHub y
   archivarlo con Zenodo para obtener DOI.
3. Ejecutar:

```powershell
python -m py_compile scripts\inspect_xdf.py scripts\auditory_event_roles.py scripts\classify_eeg_features.py scripts\make_synthetic_xdf.py scripts\plot_xdf_eeg_markers.py scripts\plot_xdf_window.py
python tests\test_event_taxonomy.py
python tests\test_classify_eeg_features.py
python tests\test_synthetic_xdf_loads.py
```

4. Repetir búsqueda de privacidad:

```powershell
Get-ChildItem -Recurse -File | Select-String -Pattern 'NOMBRE_PRIVADO|RUTA_PRIVADA|FECHA_REAL'
```

5. Revisar manualmente el README renderizado en GitHub.

## Nombre sugerido del repositorio

- `low-density-eeg-visual-auditory-bci-pilot`
- `bci-visual-auditory-attention-eeg-pipeline`

## Alcance honesto del repositorio

Este repositorio documenta el flujo y permite ejecutar componentes
genéricos del análisis. No pretende permitir una reproducción completa del
paper sin los datos crudos privados. La reproducción completa requiere los
XDF/logs locales bajo acuerdo de acceso y anonimización adicional.
