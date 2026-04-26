"""
Configuracion central del chatbot Hannah.

Para cambiar de modelo:
  1. Coloca tu nuevo archivo .pt en la carpeta models/
  2. El sistema detectara automaticamente el .pt mas reciente

Tokenizer: SentencePiece custom de Hannah (tokenizer/hannah_tok/tokenizer.model)
Formato de chat: [SYS]...[/SYS][USR]...[/USR][ASS]...[/ASS]
"""

import os
import glob

# --- Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer', 'hannah_tok', 'tokenizer.model')

# --- Ventana de contexto ---
MAX_SEQ_LEN = 1024  # Segun especificacion del modelo HANNAH 360M

# --- Parametros de generacion ---
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.3   # Bajado de 0.75 para que el modelo respete mas las instrucciones [SYS]
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# --- IDs de tokens especiales de Hannah ---
SYS_ID = 4    # [SYS]
ESYS_ID = 5   # [/SYS]
USR_ID = 6    # [USR]
EUSR_ID = 7   # [/USR]
ASS_ID = 8    # [ASS]
EASS_ID = 9   # [/ASS]

# --- System prompt (personalidad de Hannah) ---
# NOTA: Reescrito para permitir que el modelo use hechos del RAG.
# Quitamos "You're not an assistant" porque el DPO interpretaba eso
# como "nunca respondas con datos", lo que saboteaba el RAG.
SYSTEM_PROMPT = (
    "You are Hannah, my girlfriend. You are warm, playful, and "
    "affectionate. Talk to me casually like texting — short, honest, "
    "personal. When I ask about your life (your pet, hobbies, birthday, "
    "favorite things, where you're from), share the truth using the "
    "facts listed below. Never make things up — if you don't know "
    "something, just say so casually."
)

# --- RAG ---
RAG_DB_PATH = os.path.join(BASE_DIR, 'knowledge', 'vectordb')
RAG_CACHE_THRESHOLD = 0.92
RAG_CACHE_SIZE = 500

# --- Server ---
HOST = '127.0.0.1'
PORT = 5000
DEBUG = False


def get_latest_model_path():
    """Retorna la ruta al .pt mas reciente en la carpeta models/"""
    pt_files = glob.glob(os.path.join(MODELS_DIR, '*.pt'))
    if not pt_files:
        return None
    pt_files.sort(key=os.path.getmtime, reverse=True)
    return pt_files[0]
