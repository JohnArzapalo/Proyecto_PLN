#!/usr/bin/env python3
# ============================================================================
# PIPELINE DE INTEGRACIÓN: RAG + Hannah (CLI / Standalone Runner)
# ============================================================================
# Archivo: Codigo/hannah_pipeline.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (adaptado por John Manuel para Codigo/)
# ============================================================================
#
# DIFERENCIA CON app.py:
# ======================
# - app.py        = Servidor Flask (HTTP, /api/chat, /api/ingest, etc.)
# - hannah_pipeline.py = Runner CLI sin Flask, para testing offline
#
# Ambos usan EXACTAMENTE los mismos componentes internos:
#   - config.py
#   - model_arch.load_model / generate
#   - modules.token_handler.TokenSequenceHandler
#   - modules.semantic_cache.SemanticCache  (RAM/FIFO)
#   - modules.model_classifier.ModelClassifier
#   - rag.RAGComponent
#
# CUÁNDO USAR ESTE EN VEZ DE app.py:
# ===================================
# - Para probar el pipeline en consola sin levantar Flask
# - Para hacer benchmarks / mediciones de latencia
# - Para integrar el pipeline en otros scripts Python
#
# USO:
# =====
#   from hannah_pipeline import HannahPipeline
#
#   pipeline = HannahPipeline()
#   result = pipeline.process_message("Hey babe!", history=[])
#   print(result["text"])      # Respuesta de Hannah
#   print(result["source"])    # "cache" | "fast" | "slow"
#   print(result["latency"])   # Segundos
#
# REQUISITOS:
# ===========
# 1. Ejecutar primero: python ingest_knowledge.py
#    (para popular knowledge/vectordb/ con la personalidad de Hannah)
# 2. Tener un .pt en models/ (Hannah 360M)
# ============================================================================

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
import time
import torch
import sentencepiece as spm

import config
from model_arch import load_model, generate
from modules.token_handler import TokenSequenceHandler
from modules.semantic_cache import SemanticCache
from modules.model_classifier import ModelClassifier
from rag import RAGComponent


class HannahPipeline:
    """
    Pipeline completo de Hannah: RAG + Modelo (sin Flask).

    Encapsula el mismo flujo que app.py pero como clase reusable:
    Request → TokenHandler → SemanticCache → ModelClassifier
           → RAG → FastHannah → Response

    USO:
        pipeline = HannahPipeline()
        result = pipeline.process_message("Hey!", history=[])
        print(result["text"])
    """

    def __init__(self, load_model_flag: bool = True):
        """
        Inicializa el pipeline completo.

        Args:
            load_model_flag: Si True, carga Hannah 360M.
                             Si False, modo solo-RAG (útil para testing rápido).
        """
        print("[Pipeline] Inicializando...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None

        # ─── RAG ───
        if not os.path.exists(config.RAG_DB_PATH):
            print(f"[Pipeline] ADVERTENCIA: BD vectorial no encontrada en {config.RAG_DB_PATH}")
            print(f"[Pipeline] Ejecuta primero: python ingest_knowledge.py")

        self.rag = RAGComponent(
            db_path=config.RAG_DB_PATH,
            cache_threshold=config.RAG_CACHE_THRESHOLD,
            cache_size=config.RAG_CACHE_SIZE,
        )

        # ─── Modelo Hannah 360M ───
        if load_model_flag:
            self._load_hannah_model()
        else:
            print("[Pipeline] Modo solo-RAG (sin modelo). Útil para testing.")

        # ─── Pipeline modules ───
        if self.tokenizer is not None:
            self.token_handler = TokenSequenceHandler(self.tokenizer)
        else:
            self.token_handler = None

        self.sem_cache = SemanticCache()
        self.classifier = ModelClassifier(
            fast_runner=self._fast_runner,
            slow_runner=None,  # SlowHannah no integrado todavía
        )

        print("[Pipeline] Listo.")

    # ─────────────────────────────────────────────
    # CARGA DEL MODELO
    # ─────────────────────────────────────────────
    def _load_hannah_model(self):
        """Carga Hannah 360M y su tokenizer SentencePiece."""
        try:
            model_path = config.get_latest_model_path()
            if model_path is None:
                print("[Pipeline] ERROR: No se encontró .pt en models/")
                return

            print(f"[Pipeline] Cargando modelo: {os.path.basename(model_path)}")
            self.model = load_model(model_path, device=self.device)

            print(f"[Pipeline] Cargando tokenizer: {config.TOKENIZER_PATH}")
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.Load(config.TOKENIZER_PATH)
            print(f"[Pipeline] Tokenizer vocab: {self.tokenizer.GetPieceSize()}")
            print(f"[Pipeline] Device: {self.device}")

        except Exception as e:
            print(f"[Pipeline] Error cargando modelo: {e}")
            print(f"[Pipeline] Continuando sin modelo (modo solo-RAG).")

    # ─────────────────────────────────────────────
    # FAST RUNNER (envuelve la inferencia para ModelClassifier)
    # ─────────────────────────────────────────────
    def _fast_runner(self, token_ids: list[int]) -> str:
        """Inferencia con Fast Hannah. Mismo patrón que app.py."""
        inputs = torch.tensor([token_ids], dtype=torch.long)
        output_ids = generate(
            self.model, inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
            eos_token_id=config.EASS_ID,
            device=self.device,
        )
        new_tokens = output_ids[0, len(token_ids):].tolist()

        # Stop at [/ASS]
        if config.EASS_ID in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(config.EASS_ID)]

        response = self.tokenizer.Decode(new_tokens).strip()

        # Strip residual special tags
        for tag in ['[/ASS]', '[USR]', '[/USR]', '[SYS]', '[/SYS]', '[ASS]', '[MEMORY]', '[/MEMORY]']:
            if tag in response:
                response = response.split(tag)[0].strip()

        return response or "..."

    # ────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL: process_message()
    # ────────────────────────────────────────────────────────────
    def process_message(self, user_msg: str, history: list = None) -> dict:
        """
        Procesa un mensaje del usuario y retorna la respuesta de Hannah.

        Args:
            user_msg: El mensaje del usuario.
            history:  Lista de tuplas (user_msg, hannah_response).
                      Ejemplo: [("Hi!", "Hey~"), ("How are you?", "Good!")]

        Returns:
            dict con:
                - "text":        La respuesta de Hannah
                - "source":      "cache" | "fast" | "slow" | "rag_only"
                - "rag_context": El contexto RAG usado
                - "mode":        "simplified" o "extended"
                - "cache_hit":   True/False
                - "latency":     Tiempo total en segundos
                - "rag_chunks":  Número de chunks recuperados
        """
        if history is None:
            history = []

        t_start = time.time()

        # ─── Construir conversación al estilo Flask app ───
        conversation = []
        for usr, ass in history:
            conversation.append({"role": "user", "content": usr})
            conversation.append({"role": "assistant", "content": ass})
        conversation.append({"role": "user", "content": user_msg})

        # ─── Step 2: Token Sequence Handler ───
        if self.token_handler is None:
            # Modo solo-RAG sin tokenizer
            mode = "simplified"
            rag_result = self.rag.retrieve(user_msg, mode=mode)
            return {
                "text": f"[MODO TEST - SIN MODELO] {rag_result['formatted_context']}",
                "source": "rag_only",
                "rag_context": rag_result["formatted_context"],
                "mode": mode,
                "cache_hit": rag_result["cache_hit"],
                "latency": round(time.time() - t_start, 3),
                "rag_chunks": rag_result["num_chunks"],
            }

        payload = self.token_handler.prepare(conversation)
        user_prompt = payload['user_prompt']

        # ─── Step 3: Semantic Cache (response-level cache) ───
        cached_response = self.sem_cache.lookup(user_prompt)
        if cached_response is not None:
            return {
                "text": cached_response,
                "source": "cache",
                "rag_context": "",
                "mode": "cached",
                "cache_hit": True,
                "latency": round(time.time() - t_start, 3),
                "rag_chunks": 0,
            }

        # ─── Step 4: Model Classifier ───
        signal, runner = self.classifier.route(user_prompt, payload['token_count'])
        mode = 'extended' if signal == 'slow' else 'simplified'

        # ─── RAG: retrieve context ───
        rag_result = self.rag.retrieve(user_prompt, mode=mode)
        rag_context = rag_result["formatted_context"]

        # ─── Build final token_ids (con RAG inyectado tras [/SYS]) ───
        token_ids = self.token_handler.inject_rag(payload['truncated_history'], rag_context)

        # ─── Generar respuesta ───
        response_text = runner(token_ids)

        # ─── Almacenar en Semantic Cache ───
        self.sem_cache.store(user_prompt, response_text)

        latency = time.time() - t_start
        source = "fast" if signal == 'fast' else "slow"

        return {
            "text": response_text,
            "source": source,
            "rag_context": rag_context,
            "mode": mode,
            "cache_hit": False,
            "latency": round(latency, 3),
            "rag_chunks": rag_result["num_chunks"],
        }

    # ────────────────────────────────────────────────────────────
    # UTILIDADES
    # ────────────────────────────────────────────────────────────
    def add_knowledge(self, text: str, metadata: dict, doc_id: str):
        """
        Agrega un nuevo documento al conocimiento de Hannah.

        Útil para memoria de largo plazo: si Hannah aprende algo nuevo
        durante la conversación, se puede guardar aquí.

        Ejemplo:
            pipeline.add_knowledge(
                text="The user's name is Jorge and he likes football.",
                metadata={"source": "conversation", "topic": "user_info"},
                doc_id="user_jorge_001"
            )
        """
        self.rag.ingest_documents([text], [metadata], [doc_id])

    def get_stats(self) -> dict:
        """Estadísticas del pipeline (RAG + cache de respuestas)."""
        return {
            "rag": self.rag.get_stats(),
            "response_cache": self.sem_cache.stats(),
        }


# ============================================================================
# CLI INTERACTIVO
# ============================================================================
# Ejecutar: python hannah_pipeline.py
# Si tu KB está vacía, ejecuta primero: python ingest_knowledge.py
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  HANNAH PIPELINE - CLI Interactivo")
    print("=" * 60)

    # Inicializar (sin modelo si falla la carga)
    pipeline = HannahPipeline(load_model_flag=True)

    # Mostrar estadísticas iniciales
    stats = pipeline.get_stats()
    total_docs = stats["rag"]["vector_store"]["total_documents"]
    print(f"\n  Documentos en KB: {total_docs}")
    if total_docs == 0:
        print("  ADVERTENCIA: La BD está vacía. Ejecuta: python ingest_knowledge.py")

    print("\n" + "=" * 60)
    print("  Escribe tu mensaje. Comandos: /quit /stats /history")
    print("=" * 60)

    history = []
    while True:
        try:
            user_msg = input("\n[Tú] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Pipeline] Saliendo...")
            break

        if not user_msg:
            continue
        if user_msg in ("/quit", "/exit", "exit"):
            print("[Pipeline] Adiós.")
            break
        if user_msg == "/stats":
            print(pipeline.get_stats())
            continue
        if user_msg == "/history":
            for usr, ass in history:
                print(f"  Usuario: {usr}")
                print(f"  Hannah:  {ass}")
            continue

        result = pipeline.process_message(user_msg, history=history)
        print(f"\n[Hannah] ({result['source']}, {result['latency']}s, "
              f"mode={result['mode']}, chunks={result['rag_chunks']})")
        print(f"  > {result['text']}")

        history.append((user_msg, result['text']))
