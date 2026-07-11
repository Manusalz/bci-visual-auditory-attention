# Taxonomia de eventos auditivos

Esta taxonomia evita una ambiguedad importante: en este pipeline,
`standard` no significa "todo lo no atendido". `standard` es el estimulo
estandar/no objetivo. Los objetivos fisicamente presentes pero pertenecientes
a la corriente o clase no indicada se codifican por separado como
`target_ignorado`.

En el codigo se conservan nombres internos en ingles (`target`, `hit`,
`miss`) para mantener compatibilidad con los CSV y scripts. En tablas,
figuras y texto del manuscrito se recomienda esta convencion:

- Primera mencion: "estimulo objetivo (target)"; luego, "objetivo".
- `hit`: "acierto (hit)" en la primera mencion; luego, "acierto".
- `miss`: "omision (miss)" en la primera mencion; luego, "omision".
- `standard` o `standard/non-target`: "estandar/no objetivo".
- Evitar "target-hit" en texto publico; usar "ensayos con objetivo atendido y respuesta correcta".

## Analisis estricto

| Clase interna | Termino recomendado | Que ocurre | Respuesta esperada | Uso |
|---|---|---|---|---|
| `standard` | estandar/no objetivo | Sonido comun/no objetivo, izquierdo o derecho | No | Linea de base del paradigma auditivo |
| `target_atendido` | objetivo atendido | Objetivo fisico que coincide con la consigna del bloque | Si | Evento objetivo definido por la tarea |
| `target_atendido_hit` | objetivo atendido con respuesta / acierto | `target_atendido` con respuesta dentro de la ventana conductual | Si, y respondio | ERP asociado a deteccion/decision/respuesta |
| `target_atendido_miss` | objetivo atendido sin respuesta / omision | `target_atendido` sin respuesta dentro de la ventana | Si, pero no respondio | Auditoria de conducta; no fue la curva principal |
| `target_ignorado` | objetivo no atendido | Objetivo fisico del lado/tipo no indicado por la consigna | No | Control de objetivo presente pero no seleccionado |

Ejemplo: modulo de audio de dos corrientes con consigna `attL`.

- Tono comun izquierdo o derecho: `standard`.
- Objetivo izquierdo: `target_atendido`.
- Objetivo izquierdo respondido: `target_atendido_hit`.
- Objetivo derecho: `target_ignorado`.

Ejemplo: modulo de audio de cuatro clases con consigna `left_low`.

- `L/tgt_low`: `target_atendido`.
- `L/tgt_high`, `R/tgt_low` o `R/tgt_high`: `target_ignorado`.
- Cualquier `std`: `standard`.

## Analisis subjetivo/post hoc

El analisis subjetivo conserva el analisis estricto, pero agrega una
lectura exploratoria por bloque cuando la matriz conductual sugiere que el
participante siguio otra regla, por ejemplo inversion de lado o confusion
entre tonos high/low.

En esa lectura, `target_subjetivo_hit` significa:

1. evento de una clase inferida como objetivo subjetivo a partir de la
   conducta; y
2. respuesta con espacio dentro de la ventana.

No debe describirse como analisis confirmatorio. Debe reportarse como
sensibilidad o post hoc conductual.
