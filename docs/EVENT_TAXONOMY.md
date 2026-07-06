# Taxonomía de eventos auditivos

Esta taxonomía evita una ambigüedad importante: en este pipeline,
`standard` no significa "todo lo no atendido". `standard` es el sonido
común o no-target. Los targets físicamente presentes pero pertenecientes
al stream o clase no indicada se codifican por separado como
`target_ignorado`.

## Análisis estricto

| Clase | Qué ocurre | Respuesta esperada | Uso |
|---|---|---|---|
| `standard` | Sonido común/no-target, izquierdo o derecho | No | Línea de base del paradigma auditivo |
| `target_atendido` | Target físico que coincide con la consigna del bloque | Sí | Evento objetivo definido por la tarea |
| `target_atendido_hit` | `target_atendido` con respuesta dentro de la ventana conductual | Sí, y respondió | ERP/P300 asociado a detección/decisión/respuesta |
| `target_atendido_miss` | `target_atendido` sin respuesta dentro de la ventana | Sí, pero no respondió | Auditoría de conducta; no fue la curva principal |
| `target_ignorado` | Target físico del lado/tipo no indicado por la consigna | No | Control de target presente pero no seleccionado |

Ejemplo: bloque audio 2 con consigna `attL`.

- Tono común izquierdo o derecho: `standard`.
- Target izquierdo: `target_atendido`.
- Target izquierdo respondido: `target_atendido_hit`.
- Target derecho: `target_ignorado`.

Ejemplo: bloque audio 4 con consigna `left_low`.

- `L/tgt_low`: `target_atendido`.
- `L/tgt_high`, `R/tgt_low` o `R/tgt_high`: `target_ignorado`.
- Cualquier `std`: `standard`.

## Análisis subjetivo/post hoc

El análisis subjetivo conserva el análisis estricto, pero agrega una
lectura exploratoria por bloque cuando la matriz conductual sugiere que el
participante siguió otra regla, por ejemplo inversión de lado o confusión
entre tonos high/low.

En esa lectura, `target_subjetivo_hit` significa:

1. evento de una clase inferida como objetivo subjetivo a partir de la
   conducta; y
2. respuesta con espacio dentro de la ventana.

No debe describirse como análisis confirmatorio. Debe reportarse como
sensibilidad o post hoc conductual.
