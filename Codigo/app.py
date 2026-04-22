"""
Servidor Flask para el chatbot Hannah.

Pipeline completo:
  Request → [2] TokenHandler → [3] SemanticCache → [4] ModelClassifier
         → RAGComponent → FastHannah → Response

Para cambiar de modelo: coloca el nuevo .pt en models/ y reinicia.
Para integrar SlowHannah: pasa slow_runner=... al ModelClassifier.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
import torch
import sentencepiece as spm
from flask import Flask, render_template, request, jsonify

import config
from model_arch import load_model, generate
from modules.token_handler import TokenSequenceHandler
from modules.semantic_cache import SemanticCache
from modules.model_classifier import ModelClassifier
from rag import RAGComponent

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model      = None
tokenizer  = None
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

token_handler = None
sem_cache     = None
classifier    = None
rag           = None


# ---------------------------------------------------------------------------
# Runner (wraps model inference into a callable for ModelClassifier)
# ---------------------------------------------------------------------------
def _fast_runner(token_ids: list[int]) -> str:
    """Runs Fast Hannah inference and returns the decoded response string."""
    inputs     = torch.tensor([token_ids], dtype=torch.long)
    output_ids = generate(
        model, inputs,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        top_k=config.TOP_K,
        eos_token_id=config.EASS_ID,
        device=device,
    )
    new_tokens = output_ids[0, len(token_ids):].tolist()

    # Stop at [/ASS] token
    if config.EASS_ID in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(config.EASS_ID)]

    response = tokenizer.Decode(new_tokens).strip()

    # Strip any residual special tags
    for tag in ['[/ASS]', '[USR]', '[/USR]', '[SYS]', '[/SYS]', '[ASS]', '[MEMORY]', '[/MEMORY]']:
        if tag in response:
            response = response.split(tag)[0].strip()

    return response or "..."


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def init_model():
    global model, tokenizer, token_handler, sem_cache, classifier, rag

    # ── Model ──────────────────────────────────────────────────────────────
    model_path = config.get_latest_model_path()
    if model_path is None:
        print("ERROR: No .pt file found in models/")
        return False

    print(f"Loading model: {os.path.basename(model_path)}")
    model = load_model(model_path, device=device)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {config.TOKENIZER_PATH}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(config.TOKENIZER_PATH)
    print(f"Tokenizer vocab: {tokenizer.GetPieceSize()}")

    # ── Pipeline modules ────────────────────────────────────────────────────
    print("Initializing pipeline modules...")
    token_handler = TokenSequenceHandler(tokenizer)
    sem_cache     = SemanticCache()
    classifier    = ModelClassifier(fast_runner=_fast_runner, slow_runner=None)

    # ── RAG ─────────────────────────────────────────────────────────────────
    print(f"Initializing RAG (db: {config.RAG_DB_PATH})...")
    rag = RAGComponent(
        db_path=config.RAG_DB_PATH,
        cache_threshold=config.RAG_CACHE_THRESHOLD,
        cache_size=config.RAG_CACHE_SIZE,
    )

    print(f"Device: {device}")
    print("Server ready!")
    return True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    if model is None:
        return jsonify({'error': 'Model not loaded. Place a .pt in models/'}), 503

    data         = request.json
    conversation = data.get('conversation', [])
    if not conversation:
        return jsonify({'error': 'Empty conversation'}), 400

    # ── Step 2: Token Sequence Handler ─────────────────────────────────────
    payload     = token_handler.prepare(conversation)
    user_prompt = payload['user_prompt']

    # ── Step 3: Semantic Cache lookup ───────────────────────────────────────
    cached = sem_cache.lookup(user_prompt)
    if cached is not None:
        return jsonify({'response': cached, 'source': 'cache'})

    # ── Step 4: Model Classifier ────────────────────────────────────────────
    signal, runner = classifier.route(user_prompt, payload['token_count'])
    mode = 'extended' if signal == 'slow' else 'simplified'

    # ── RAG: retrieve context ───────────────────────────────────────────────
    rag_result  = rag.retrieve(user_prompt, mode=mode)
    rag_context = rag_result.get('formatted_context', '')

    # ── Build final token_ids (with RAG context injected after [/SYS]) ──────
    token_ids = token_handler.inject_rag(payload['truncated_history'], rag_context)

    # ── Generate ────────────────────────────────────────────────────────────
    response = runner(token_ids)

    # ── Store in Semantic Cache ─────────────────────────────────────────────
    sem_cache.store(user_prompt, response)

    return jsonify({'response': response})


@app.route('/api/model-info')
def model_info():
    if model is None:
        return jsonify({'loaded': False})

    model_path  = config.get_latest_model_path()
    rag_stats   = rag.get_stats() if rag else {}
    cache_stats = sem_cache.stats() if sem_cache else {}

    return jsonify({
        'loaded':      True,
        'model_file':  os.path.basename(model_path) if model_path else None,
        'device':      device,
        'parameters':  f"{sum(p.numel() for p in model.parameters()):,}",
        'rag':         rag_stats,
        'cache':       cache_stats,
    })


@app.route('/api/ingest', methods=['POST'])
def ingest():
    """
    Endpoint to add documents to the RAG knowledge base.
    Body: { "documents": [...], "metadatas": [...], "ids": [...] }
    """
    if rag is None:
        return jsonify({'error': 'RAG not initialized'}), 503

    data = request.json
    docs      = data.get('documents', [])
    metadatas = data.get('metadatas', [{}] * len(docs))
    ids       = data.get('ids', [f"doc_{i}" for i in range(len(docs))])

    if not docs:
        return jsonify({'error': 'No documents provided'}), 400

    rag.ingest_documents(docs, metadatas, ids)
    stats = rag.get_stats()
    return jsonify({
        'ingested': len(docs),
        'total_documents': stats['vector_store']['total_documents'],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    init_model()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, use_reloader=False)
