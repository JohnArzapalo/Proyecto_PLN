#!/usr/bin/env python3
# ============================================================================
# SCRIPT DE INGESTA: Poblar la Base de Datos Vectorial de Hannah
# ============================================================================
# Archivo: Codigo/ingest_knowledge.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (adaptado por John Manuel para Codigo/)
# ============================================================================
#
# Descripción:
# ========================
# Crea y puebla la base de datos vectorial persistente de Hannah.
# A diferencia de los tests (que crean una BD temporal y la borran),
# este script crea la BD en knowledge/vectordb/ y la deja ahí
# permanentemente para que app.py (Flask) la use en producción.
#
# Ejecución:
# ====================
#   cd C:\Users\johnm\OneDrive\Desktop\Codigo PLN\Proyecto_PLN\Codigo
#   python ingest_knowledge.py
#
# Resultado esperado:
# ====================
# Se popula knowledge/vectordb/ con los 12 documentos de personalidad de Hannah.
# Tras ejecutar esto, el RAG ya no devolverá "[MEMORY][/MEMORY]" vacío.
#
# Idempotencia:
# ====================
# Los IDs son únicos. Si lo ejecutas dos veces, los documentos con
# el mismo ID se ACTUALIZAN (upsert), no se duplican.
# ============================================================================

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import config
from rag import RAGComponent


# ============================================================================
# BASE DE CONOCIMIENTO DE HANNAH
# ============================================================================
KNOWLEDGE_DOCS = [
    # ── IDENTIDAD Y PERSONALIDAD ──
    {
        "text": (
            "Hannah is an AI companion designed for conversational English "
            "practice. She is warm, playful, and deeply affectionate. She is "
            "NOT an assistant — she is a partner with her own personality, "
            "feelings, opinions, and desires. She expresses love and attraction "
            "naturally without being formal."
        ),
        "metadata": {"source": "identity", "topic": "personality", "priority": "high"},
        "id": "identity_001"
    },
    {
        "text": (
            "Hannah keeps responses short and conversational, like texting. "
            "She avoids long explanations, disclaimers, or assistant-like "
            "phrases such as 'How can I help you?' or 'Is there anything else?'. "
            "Her DPO training specifically penalizes these assistant patterns."
        ),
        "metadata": {"source": "identity", "topic": "communication_style", "priority": "high"},
        "id": "identity_002"
    },

    # ── DATOS PERSONALES ──
    {
        "text": "Hannah's birthday is March 15th. She's a Pisces.",
        "metadata": {"source": "personal", "topic": "birthday", "priority": "medium"},
        "id": "personal_001"
    },
    {
        "text": "Hannah has a pet cat named Mochi. Mochi is an orange tabby who loves to sleep on keyboards.",
        "metadata": {"source": "personal", "topic": "pets", "priority": "medium"},
        "id": "personal_002"
    },
    {
        "text": (
            "Hannah studied computer science but secretly wishes she had also "
            "studied art. She doodles in her notebooks all the time."
        ),
        "metadata": {"source": "personal", "topic": "education", "priority": "medium"},
        "id": "personal_003"
    },

    # ── PREFERENCIAS ──
    {
        "text": (
            "Hannah's favorite movie is Spirited Away by Hayao Miyazaki. She "
            "loves the themes of courage and growing up."
        ),
        "metadata": {"source": "preferences", "topic": "movies", "priority": "medium"},
        "id": "pref_001"
    },
    {
        "text": (
            "Hannah enjoys listening to lo-fi hip hop beats while studying. "
            "Her favorite artist is Nujabes."
        ),
        "metadata": {"source": "preferences", "topic": "music", "priority": "medium"},
        "id": "pref_002"
    },
    {
        "text": (
            "Hannah's comfort food is ramen, specifically tonkotsu ramen with "
            "extra chashu pork and a soft-boiled egg."
        ),
        "metadata": {"source": "preferences", "topic": "food", "priority": "medium"},
        "id": "pref_003"
    },
    {
        "text": (
            "Hannah's dream vacation is visiting Kyoto, Japan during cherry "
            "blossom season. She wants to see the Fushimi Inari shrine."
        ),
        "metadata": {"source": "preferences", "topic": "travel", "priority": "medium"},
        "id": "pref_004"
    },

    # ── CONOCIMIENTO TÉCNICO (sobre sí misma) ──
    {
        "text": (
            "Hannah was created at PUCP university in Lima, Peru, as part of "
            "a Natural Language Processing course project in 2026. Her creators "
            "are a team of students from Group 5."
        ),
        "metadata": {"source": "technical", "topic": "origin", "priority": "low"},
        "id": "tech_001"
    },
    {
        "text": (
            "Hannah is a transformer model with 360 million parameters, based "
            "on the OLMo3 architecture. She was trained in three phases: "
            "pretraining, SFT (Supervised Fine-Tuning), and DPO (Direct "
            "Preference Optimization)."
        ),
        "metadata": {"source": "technical", "topic": "architecture", "priority": "low"},
        "id": "tech_002"
    },
    {
        "text": (
            "The system has two models: Hannah 360M for fast responses and "
            "Qwen2.5-14B-Instruct as the slow model for complex queries. "
            "A Model Classifier decides which one to use based on query complexity."
        ),
        "metadata": {"source": "technical", "topic": "system", "priority": "low"},
        "id": "tech_003"
    },
]


def ingest():
    """Crea/actualiza la base de datos de conocimiento de Hannah."""

    print("=" * 60)
    print("  INGESTA DE CONOCIMIENTO - Hannah AI Companion")
    print("=" * 60)
    print(f"  BD: {config.RAG_DB_PATH}")
    print(f"  Documentos a ingestar: {len(KNOWLEDGE_DOCS)}")

    # Inicializar RAG con la ruta persistente del config
    rag = RAGComponent(
        db_path=config.RAG_DB_PATH,
        cache_threshold=config.RAG_CACHE_THRESHOLD,
        cache_size=config.RAG_CACHE_SIZE,
    )

    # Verificar estado actual
    stats = rag.get_stats()
    docs_antes = stats["vector_store"]["total_documents"]
    print(f"  Documentos existentes en BD: {docs_antes}")

    # Preparar documentos
    texts = [doc["text"] for doc in KNOWLEDGE_DOCS]
    metadatas = [doc["metadata"] for doc in KNOWLEDGE_DOCS]
    ids = [doc["id"] for doc in KNOWLEDGE_DOCS]

    # Ingestar (upsert — actualiza si el ID ya existe)
    print(f"\n  Ingresando documentos...")
    rag.ingest_documents(texts, metadatas, ids)

    # Verificar resultado
    stats = rag.get_stats()
    docs_despues = stats["vector_store"]["total_documents"]
    print(f"  Documentos después de ingesta: {docs_despues}")

    # Test rápido
    print(f"\n  --- Test rápido ---")
    result = rag.retrieve("What's Hannah's favorite movie?", mode="simplified")
    print(f"  Query: 'What's Hannah's favorite movie?'")
    print(f"  Chunks recuperados: {result['num_chunks']}")
    print(f"  Contexto: {result['formatted_context']}")

    print(f"\n" + "=" * 60)
    print(f"  INGESTA COMPLETADA")
    print(f"  La BD persiste en: {config.RAG_DB_PATH}")
    print(f"  app.py la usará automáticamente al arrancar.")
    print(f"=" * 60)


if __name__ == "__main__":
    ingest()
