"""
Servidor Flask para el chatbot Hannah.
Conecta el frontend con el modelo .pt usando el formato exacto
de tokens especiales: [SYS]...[/SYS][USR]...[/USR][ASS]...[/ASS]
"""

import os
import torch
import sentencepiece as spm
from flask import Flask, render_template, request, jsonify

import config
from model_arch import load_model, generate

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# --- Estado global ---
model = None
tokenizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_model():
    """Carga el modelo y tokenizer al iniciar el servidor."""
    global model, tokenizer

    model_path = config.get_latest_model_path()
    if model_path is None:
        print("=" * 60)
        print("ERROR: No se encontro ningun archivo .pt en la carpeta models/")
        print(f"Coloca tu modelo en: {config.MODELS_DIR}")
        print("=" * 60)
        return False

    print(f"Cargando modelo: {os.path.basename(model_path)}")
    model = load_model(model_path, device=device)

    print(f"Cargando tokenizer: {config.TOKENIZER_PATH}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(config.TOKENIZER_PATH)
    print(f"Tokenizer vocab: {tokenizer.GetPieceSize()}")

    print(f"Dispositivo: {device}")
    print("Servidor listo!")
    return True


def format_prompt(conversation_history):
    """
    Construye el prompt con tokens especiales de Hannah:
    [SYS] system prompt [/SYS][USR] msg [/USR][ASS] resp [/ASS]...
    """
    prompt = f"[SYS] {config.SYSTEM_PROMPT} [/SYS]"
    for msg in conversation_history:
        if msg['role'] == 'user':
            prompt += f"[USR] {msg['content']} [/USR]"
        else:
            prompt += f"[ASS] {msg['content']} [/ASS]"
    # Abrir tag [ASS] para que el modelo genere la respuesta
    prompt += "[ASS]"
    return prompt


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    if model is None:
        return jsonify({'error': 'Modelo no cargado. Coloca un .pt en models/'}), 503

    data = request.json
    conversation = data.get('conversation', [])

    if not conversation:
        return jsonify({'error': 'Conversacion vacia'}), 400

    # Construir prompt con tokens especiales
    prompt = format_prompt(conversation)
    import sys; print(f"[DEBUG] Prompt: {prompt[:120]}...", flush=True)

    # Tokenizar con SentencePiece
    token_ids = tokenizer.Encode(prompt)
    print(f"[DEBUG] Tokens: {len(token_ids)}, first 10: {token_ids[:10]}", flush=True)

    # Truncar historial si excede la ventana de contexto
    while len(conversation) > 2 and len(token_ids) > config.MAX_SEQ_LEN - config.MAX_NEW_TOKENS:
        conversation.pop(0)
        prompt = format_prompt(conversation)
        token_ids = tokenizer.Encode(prompt)

    inputs = torch.tensor([token_ids], dtype=torch.long)

    # Generar respuesta - stop en [/ASS] (id=9)
    output_ids = generate(
        model, inputs,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        top_k=config.TOP_K,
        eos_token_id=config.EASS_ID,
        device=device,
    )

    # Decodificar solo los tokens nuevos
    new_tokens = output_ids[0, len(token_ids):].tolist()
    print(f"[DEBUG] Generated {len(new_tokens)} tokens, first 10: {new_tokens[:10]}", flush=True)

    # Remover [/ASS] si esta presente
    if config.EASS_ID in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(config.EASS_ID)]

    response = tokenizer.Decode(new_tokens).strip()

    # Limpiar respuesta si contiene tags residuales
    for tag in ['[/ASS]', '[USR]', '[/USR]', '[SYS]', '[/SYS]', '[ASS]']:
        if tag in response:
            response = response.split(tag)[0].strip()

    if not response:
        response = "..."

    return jsonify({'response': response})


@app.route('/api/model-info')
def model_info():
    if model is None:
        return jsonify({'loaded': False, 'error': 'Sin modelo'})
    model_path = config.get_latest_model_path()
    return jsonify({
        'loaded': True,
        'model_file': os.path.basename(model_path) if model_path else None,
        'device': device,
        'parameters': f"{sum(p.numel() for p in model.parameters()):,}",
    })


if __name__ == '__main__':
    init_model()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, use_reloader=False)
