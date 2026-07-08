# Clasificación y validación

La métrica principal del paquete es la exactitud balanceada
(`balanced accuracy`, BA), definida como el promedio de sensibilidad por
clase. Es preferible a la exactitud simple cuando las clases tienen tamaños
distintos.

Para cada contraste se recomienda reportar:

- archivo/corrida de origen desidentificado;
- contraste exacto;
- clase positiva y clase negativa;
- cantidad de épocas por clase;
- hits, misses y falsas alarmas conductuales;
- tipo de análisis: principal, sensibilidad o post hoc;
- ventana temporal o banda de frecuencia;
- accuracy simple y balanced accuracy;
- intervalo por bootstrap;
- p empírica por permutación;
- cantidad de particiones de validación cruzada.

La p de permutación se estima reetiquetando al azar las clases y
recalculando BA. En estudios piloto pequeños, esta prueba ayuda a no
sobreinterpretar una BA aparentemente alta.

Para analisis complementarios orientados a P300 se puede usar una validacion
mas exigente por bloque/run, cuando el numero de bloques lo permite. En ese
caso, cada fold deja afuera un bloque completo, el escalado, xDAWN,
covarianzas y clasificador se ajustan solo con el train, y la permutacion se
hace dentro de bloque/run para preservar la estructura temporal del registro.
