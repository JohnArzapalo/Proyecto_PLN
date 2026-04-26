# Estado del Sistema RAG - Hannah AI Companion

**Proyecto**: Hannah 360M - PUCP NLP Course - Group 5
**Fecha del documento**: 2026-04-26
**Autor**: John Manuel Arzapalo Arana

---

## 1. Resumen Ejecutivo

El sistema RAG está **completamente operativo a nivel de infraestructura**: recupera, almacena, persiste y entrega contexto correcto al modelo. Sin embargo, el modelo Fast (Hannah 360M) **no consigue usar los hechos inyectados**, alucinando respuestas. Esta es una limitación del modelo, no del pipeline.

| Capa | Estado |
|---|---|
| RAG (recuperación, embeddings, ChromaDB) | OPERATIVA |
| Semantic Cache (RAM/FIFO) | OPERATIVA |
| Model Classifier (selector fast/slow) | OPERATIVO con hook para SlowHannah |
| Token Handler + inyección al modelo | OPERATIVO |
| Hannah 360M usando contexto del RAG | **FALLANDO** (ver Sección 7) |
| SlowHannah (Qwen2.5-14B-Instruct) | NO INTEGRADO (pendiente) |

---

## 2. Arquitectura Implementada

### 2.1 Pipeline Completo

```
Usuario
  |
  v
Flask App (app.py)
  |
  v
TokenSequenceHandler (modules/token_handler.py)
  - Formatea conversacion con [SYS]/[USR]/[ASS]
  - Trunca historial si excede contexto
  |
  v
SemanticCache (modules/semantic_cache.py)
  - RAM, 500 entradas, FIFO, threshold 0.92
  - HIT -> retorna cacheado
  - MISS -> continua
  |
  v
ModelClassifier (modules/model_classifier.py)
  - Heuristica: tokens > 80 o keywords complejos -> 'slow'
  - Por ahora ambas señales rutean a FastHannah
  - Hook listo para SlowHannah
  |
  v
RAGComponent (rag/)
  - QueryEnhancer: simplificada (1 query) o extendida (5 queries con HyDE)
  - VectorStore: ChromaDB con HNSW cosine
  - ContextHandler: top-3 (simplified) o top-10 (extended) chunks
  |
  v
inject_rag() en TokenHandler
  - Inyecta hechos del RAG dentro del bloque [SYS]
  |
  v
FastHannah (Hannah 360M, model_arch.py)
  - Genera respuesta
  |
  v
Response al usuario
```

### 2.2 Estructura de Archivos del Proyecto

```
Codigo/
├── app.py                         # Servidor Flask
├── config.py                      # Configuracion central
├── model_arch.py                  # Carga de Hannah 360M (OLMo3)
├── ingest_knowledge.py            # Script para popular KB (NUEVO)
├── hannah_pipeline.py             # CLI standalone runner (NUEVO)
├── modules/
│   ├── token_handler.py           # Step 2: TokenSequenceHandler
│   ├── semantic_cache.py          # Step 3: cache RAM/FIFO de respuestas
│   └── model_classifier.py        # Step 4: selector fast/slow
├── rag/                           # Paquete RAG completo (6 modulos)
│   ├── __init__.py
│   ├── embeddings.py              # all-MiniLM-L6-v2, 384 dims
│   ├── vector_store.py            # ChromaDB con cosine HNSW
│   ├── semantic_cache.py          # cache interno del RAG
│   ├── query_enhancer.py          # QE + HyDE para modo extended
│   ├── context_handler.py         # Reranking + formato [MEMORY]
│   └── rag_component.py           # Orquestador
├── knowledge/
│   └── vectordb/                  # ChromaDB persistente (12 docs)
│       ├── chroma.sqlite3
│       └── 2c255a16-.../
│           ├── data_level0.bin    # HNSW index
│           └── header.bin
├── models/
│   └── hannah_personality_final.pt
├── tokenizer/
│   └── hannah_tok/tokenizer.model
├── static/, templates/            # Frontend Flask
└── ESTADO_RAG_HANNAH.md           # Este documento
```

---

## 3. Tokens Especiales del Tokenizer

Todos los tokens necesarios están registrados como tokens unicos en el SentencePiece:

| Token | ID | Estado |
|---|---|---|
| `[SYS]` | 4 | Registrado |
| `[/SYS]` | 5 | Registrado |
| `[USR]` | 6 | Registrado |
| `[/USR]` | 7 | Registrado |
| `[ASS]` | 8 | Registrado |
| `[/ASS]` | 9 | Registrado |
| `[MEMORY]` | 10 | Registrado, **pero modelo no lo usa** (ver Seccion 7) |
| `[/MEMORY]` | 11 | Registrado, **pero modelo no lo usa** |

**Vocabulario total**: 32,000 tokens.

---

## 4. Verificaciones Realizadas (20 preguntas)

### 4.1 Embeddings (rag/embeddings.py)
- Vectores de 384 dimensiones: SI
- Embeddings normalizados (norm = 1.0): SI
- Modelo all-MiniLM-L6-v2 cargado en CPU: SI

### 4.2 Vector Store (rag/vector_store.py)
- Insercion y recuperacion: SI
- Espacio cosine en ChromaDB: SI (`metadata={"hnsw:space": "cosine"}`)
- Persistencia entre reinicios: SI (verificado leyendo 12 docs tras matar el proceso)

### 4.3 Semantic Cache (rag/semantic_cache.py - cache interno del RAG)
- HIT con queries similares (score >= 0.92): SI
- MISS con queries diferentes: SI
- FIFO al llenar 500 entradas: codigo verificado (`pop(0)`)

### 4.4 Query Enhancer (rag/query_enhancer.py)
- Modo simplified: 1 query: SI
- Modo extended: 5 queries (original + declarativa + keywords + contextual + HyDE): SI
- HyDE template segun tipo: SI ("¿Que es NLP?" -> "El concepto de...")

### 4.5 Context Handler (rag/context_handler.py)
- Wrap en `[MEMORY]/[/MEMORY]`: SI
- Simplified < 800 chars / ~200 tokens: SI
- Extended incluye `[Fuente: X]`: SI
- Separadores: simplified=`" "` (espacio), extended=`"\n---\n"`: SI

### 4.6 RAG Completo (rag/rag_component.py)
- End-to-end simplified: 3 chunks, ~80 tokens: SI
- End-to-end extended: 8 chunks con QE: SI
- Cache hit en queries identicas: SI (sim=1.000)
- Async (aretrieve) equivalente a sync (retrieve): SI

---

## 5. Base de Conocimiento Poblada

12 documentos sobre la personalidad de Hannah, en `knowledge/vectordb/`:

| ID | Categoria | Contenido |
|---|---|---|
| identity_001 | Identidad | AI companion para conversational English |
| identity_002 | Comunicacion | Estilo casual tipo texting |
| personal_001 | Cumpleanos | March 15th, Pisces |
| personal_002 | Mascota | Cat named Mochi (orange tabby) |
| personal_003 | Educacion | Computer science |
| pref_001 | Pelicula | Spirited Away (Miyazaki) |
| pref_002 | Musica | Lo-fi hip hop, Nujabes |
| pref_003 | Comida | Tonkotsu ramen |
| pref_004 | Viaje | Kyoto durante cherry blossom |
| tech_001 | Origen | PUCP, Lima, Peru, NLP course 2026 |
| tech_002 | Arquitectura | 360M params, OLMo3, pretraining + SFT + DPO |
| tech_003 | Sistema | Fast + Slow models, ModelClassifier |

**Comando para repoblar/actualizar**:
```bash
python ingest_knowledge.py
```

---

## 6. Cambios y Optimizaciones Aplicados

### 6.1 Optimizaciones del RAG (vs version standalone)

| Cambio | Razon |
|---|---|
| Shared embedder pattern | El paquete standalone cargaba all-MiniLM-L6-v2 **3 veces** (270 MB). Ahora se carga una vez y se comparte entre VectorStore, SemanticCache, ContextHandler |
| Empty-KB guard en `retrieve()` | Sin esto, una KB vacia causaba excepcion al pedir n_results > 0 |
| Clamp de n_results a docs disponibles en `_multi_query_search` | Evita pedir mas docs de los que existen |

### 6.2 Reescritura de modules/semantic_cache.py

Era SQLite + LFU. Se reescribio a **RAM + FIFO + 500 entradas + threshold 0.92** segun especificacion de arquitectura. Almacena pares (prompt, response) y compara por dot product de embeddings normalizados.

### 6.3 Fixes de Windows / Encoding

- Caracter `→` en model_classifier.py reemplazado por `->` (Unicode crash)
- `sys.stdout.reconfigure(encoding='utf-8')` añadido en app.py, ingest_knowledge.py, hannah_pipeline.py
- Flag `-u` (unbuffered) requerido al ejecutar Python con redireccion de logs

### 6.4 Archivos Nuevos Integrados

- **ingest_knowledge.py**: script idempotente para popular la KB con los 12 documentos
- **hannah_pipeline.py**: CLI runner alternativo a Flask, usa los mismos componentes que app.py

### 6.5 Cambios para Intentar Resolver el Problema (Seccion 7)

| Cambio | Archivo | Resultado |
|---|---|---|
| Mover hechos del RAG de `[MEMORY]...[/MEMORY]` al interior del bloque `[SYS]` | modules/token_handler.py (`inject_rag`) | NO RESOLVIO |
| Bajar `TEMPERATURE` de 0.75 a 0.3 | config.py | NO RESOLVIO |
| Reescribir SYSTEM_PROMPT para permitir respuestas factuales (quitar "You're not an assistant") | config.py | NO RESOLVIO |

---

## 7. PROBLEMATICA ACTUAL

### 7.1 Sintoma

El modelo Hannah 360M alucina respuestas factuales en vez de usar los hechos del RAG correctamente recuperados.

### 7.2 Tests Realizados

**Test A** (configuracion original con `[MEMORY]...[/MEMORY]` separado):

| Pregunta | RAG dice | Respuesta del modelo |
|---|---|---|
| Do you have a pet? | Mochi (orange tabby) | "I have a cat, **Simon**, getting out of the closet" |
| Favorite movie? | Spirited Away | "The one with the few favorite people... and you" |
| When is your birthday? | March 15, Pisces | "I'm thinking of you on **Saturday**" |

**Test B** (despues de aplicar Opciones A+B: hechos en `[SYS]`, temperature 0.3, prompt mas explicito):

| Pregunta | RAG dice | Respuesta del modelo |
|---|---|---|
| Do you have a pet? | Mochi (orange tabby) | "Oh, yeah I have a **dog**!" |
| Favorite movie? | Spirited Away | "I'm a big fan of **The Shawshank Redemption**, can't wait to see you tonight, and I'm obsessed with the hanck" |
| When is your birthday? | March 15, Pisces | "I'm not sure, **maybe in 2022?**" |

### 7.3 Diagnostico de Causa Raiz

**Verificado que SI funciona correctamente:**
- ChromaDB persiste los 12 docs y los carga al iniciar
- El RAG recupera chunks relevantes (scores 0.40-0.61 para queries factuales)
- El tokenizer reconoce todos los tokens especiales
- El prompt final que recibe el modelo contiene los hechos correctos en `[SYS]`
- Verificado decodificando el `token_ids` final: los hechos estan ahi

**El problema es exclusivamente del modelo:**

1. **Token `[MEMORY]` sin entrenamiento semantico**
   El tokenizer tiene `[MEMORY]` (ID 10) y `[/MEMORY]` (ID 11) registrados, pero el entrenamiento del modelo (pretraining + SFT + DPO) **no incluyo ejemplos donde el contenido entre estos tokens deba usarse como contexto recuperado**. Para el modelo son tokens sin asociacion aprendida.

2. **DPO sobre-especializado en afecto**
   El DPO penalizo respuestas tipo asistente. El modelo desarrollo un sesgo fuerte hacia respuestas afectivas/casuales y evita responder con datos concretos, incluso cuando estan en `[SYS]`.

3. **Modelo demasiado pequeño para RAG factual**
   360M parametros es insuficiente para seguir instrucciones complejas tipo "usa los siguientes hechos para responder". Modelos como Llama-3-1B apenas pueden hacer RAG basico, y solo si fueron instruction-tuned.

4. **Auto-contaminacion por historial conversacional**
   Una vez que el modelo alucina en el turno N, sigue su propia mentira en el turno N+1. La respuesta "I'm not sure, maybe 2022?" del Test B turno 3 es evidencia: el modelo se rinde despues de mentir dos veces seguidas.

5. **La inyeccion en `[SYS]` mejoro el formato pero no la conducta**
   El modelo SI lee el `[SYS]` (lo demuestra al ser afectivo), pero **trata los hechos del RAG como sugerencias decorativas, no como datos a usar**.

### 7.4 Por Que Las Soluciones Aplicadas No Funcionaron

| Solucion intentada | Hipotesis | Resultado real |
|---|---|---|
| Inyectar hechos en `[SYS]` en lugar de `[MEMORY]` | El modelo si entrena con `[SYS]`, deberia respetarlo | Lee la personalidad pero ignora los hechos |
| Bajar temperatura a 0.3 | Menor randomness = mas adherencia a instrucciones | Sigue alucinando, ahora mas consistentemente |
| Reescribir SYSTEM_PROMPT con permiso explicito | El DPO no rechazaria datos si el system prompt los pide | DPO domina sobre el system prompt |

---

## 8. Recomendacion de Solucion Real

### 8.1 Integracion de SlowHannah (Qwen2.5-14B-Instruct)

Es la solucion arquitecturalmente correcta y ya esta prevista en el codigo:

- **ModelClassifier** ya emite señal `'slow'` correctamente para queries factuales/largos
- En `app.py` linea 99 hay un hook listo:
  ```python
  classifier = ModelClassifier(fast_runner=_fast_runner, slow_runner=None)
  ```
- Solo hay que cambiar `slow_runner=None` por `slow_runner=qwen_runner` cuando se integre
- Qwen2.5-14B es instruction-tuned, **lee y usa correctamente bloques `[MEMORY]/[/MEMORY]`** sin necesidad de inyectarlos en `[SYS]`
- Su ventana de contexto es de 32K tokens (vs 1024 de Hannah), holgada para `extended` mode

### 8.2 Plan de Trabajo Sugerido

1. **Corto plazo (esta semana)**: aceptar que Fast Hannah es para conversacion casual romantica. NO promocionar el RAG factual con Fast Hannah en la presentacion.

2. **Medio plazo**: integrar Qwen2.5-14B-Instruct como SlowHannah. Para queries factuales el ModelClassifier rutea a Slow, que SI usa el RAG correctamente. Para chat romantico/casual, Fast Hannah responde rapido.

3. **Demo del proyecto**:
   - Mostrar Fast Hannah respondiendo casual a "hi", "i miss you", "tell me a joke"
   - Mostrar Slow Hannah (Qwen) respondiendo factual con `[MEMORY]` a "what's your favorite movie?", "tell me about you"
   - Esto demuestra el valor de la arquitectura dual fast/slow

### 8.3 Alternativa de Bajo Costo

Si no se puede integrar Qwen a tiempo:

- **Limitar el dominio del chat**: la presentacion enfatiza que Hannah es una *companion para practicar ingles*, no un Q&A bot
- **Mostrar el RAG funcionando aislado**: los tests de `python -m rag.rag_component` y `python ingest_knowledge.py` demuestran que el RAG recupera bien
- **Documentar la limitacion**: incluir esta seccion 7 en el reporte final como hallazgo del proyecto (es un resultado valido y honesto)

---

## 9. Comandos Operativos

### Arrancar el servidor de produccion
```bash
cd "C:\Users\johnm\OneDrive\Desktop\Codigo PLN\Proyecto_PLN\Codigo"
python -u app.py
# Servidor en http://127.0.0.1:5000
```

### Repopular la KB (idempotente)
```bash
python ingest_knowledge.py
```

### CLI standalone (sin Flask)
```bash
python hannah_pipeline.py
```

### Verificar estado de la KB sin iniciar Flask
```bash
python -c "
import config
from rag import RAGComponent
rag = RAGComponent(db_path=config.RAG_DB_PATH)
print(rag.get_stats())
"
```

### Tests individuales de cada modulo del RAG
```bash
python -m rag.embeddings
python -m rag.vector_store
python -m rag.semantic_cache
python -m rag.query_enhancer
python -m rag.context_handler
python -m rag.rag_component
```

### Limpieza completa (procesos + cache)
```powershell
# PowerShell:
Get-Process | Where-Object { $_.ProcessName -match 'python' } | Stop-Process -Force
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

---

## 10. Configuracion Actual

| Parametro | Valor | Archivo |
|---|---|---|
| MAX_SEQ_LEN | 1024 | config.py |
| MAX_NEW_TOKENS | 200 | config.py |
| TEMPERATURE | 0.3 (bajado de 0.75) | config.py |
| TOP_K | 50 | config.py |
| TOP_P | 0.9 | config.py |
| RAG_CACHE_THRESHOLD | 0.92 | config.py |
| RAG_CACHE_SIZE | 500 | config.py |
| Embedding model | all-MiniLM-L6-v2 | rag/embeddings.py |
| Vector dim | 384 | rag/embeddings.py |
| ChromaDB metric | cosine (HNSW) | rag/vector_store.py |
| RAG simplified mode | top-3, ~200 tokens, separator=" " | rag/context_handler.py |
| RAG extended mode | top-10, ~1500 tokens, separator="\n---\n" | rag/context_handler.py |

---

## 11. Conclusion

El proyecto entrega un **pipeline RAG completo, funcional y verificado** que cumple la arquitectura especificada. La unica limitacion es que **Hannah 360M no fue entrenada con datos de RAG**, lo que la hace incapaz de usar el contexto recuperado.

Esta limitacion **no es un bug del sistema** sino una caracteristica del modelo. La arquitectura ya esta preparada para resolverla via integracion de SlowHannah (Qwen 14B), que es el siguiente paso natural del proyecto.
