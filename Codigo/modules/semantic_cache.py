"""
Step 3 — Semantic Cache (in-memory, FIFO)

Stores prompt-response pairs in RAM indexed by their normalized embedding.
On each lookup it computes the dot product against every stored entry
(equivalent to cosine similarity because the embeddings are normalized):
  - similarity >= threshold  →  HIT  (returns cached response)
  - similarity <  threshold  →  MISS (returns None, flow continues)

Capacity: max 500 entries. When full, the oldest entry is evicted (FIFO).
Lifetime: in-memory only; the cache is cleared on server restart by design.
The embedding model (all-MiniLM-L6-v2, ~90 MB) is loaded lazily on
first use so it does not slow down server startup.
"""

import numpy as np


# Module-level singleton so the embedding model is loaded only once per process
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("[SemanticCache] Loading embedding model (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[SemanticCache] Embedding model ready.")
    return _embed_model


class SemanticCache:
    """
    In-memory semantic similarity cache with FIFO eviction.

    Usage:
        cache = SemanticCache()

        result = cache.lookup(user_prompt)
        if result is not None:
            return result          # HIT

        response = model.generate(...)
        cache.store(user_prompt, response)   # MISS → store for next time
    """

    def __init__(
        self,
        threshold: float = 0.92,
        max_entries: int = 500,
    ):
        """
        Args:
            threshold:   minimum cosine similarity to count as a HIT  [0, 1]
            max_entries: maximum number of entries; oldest is evicted (FIFO) on overflow
        """
        self.threshold   = threshold
        self.max_entries = max_entries

        # In-memory storage. Each entry is a dict:
        #   {'prompt': str, 'embedding': np.ndarray (normalized), 'response': str, 'hits': int}
        self._entries: list[dict] = []
        self._total_hits: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Return a normalized embedding so dot product = cosine similarity."""
        model = _get_embed_model()
        emb = model.encode([text], normalize_embeddings=True)[0]
        return emb.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, prompt: str) -> str | None:
        """
        Searches the cache for a semantically similar prompt.

        Returns:
            Cached response string on HIT, None on MISS.
        """
        if not self._entries:
            print("[SemanticCache] MISS (cache empty)")
            return None

        query_emb = self._embed(prompt)

        best_sim = -1.0
        best_idx = -1
        for idx, entry in enumerate(self._entries):
            # Dot product of normalized vectors == cosine similarity
            sim = float(np.dot(query_emb, entry['embedding']))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= self.threshold:
            self._entries[best_idx]['hits'] += 1
            self._total_hits += 1
            print(f"[SemanticCache] HIT  (sim={best_sim:.3f})")
            return self._entries[best_idx]['response']

        print(f"[SemanticCache] MISS (best_sim={best_sim:.3f})")
        return None

    def store(self, prompt: str, response: str) -> None:
        """
        Embeds the prompt and appends the (prompt, response) pair.
        Evicts the oldest entry (FIFO) when the cache is at capacity.
        """
        # FIFO eviction: when full, drop the oldest entry (index 0)
        if len(self._entries) >= self.max_entries:
            removed = self._entries.pop(0)
            print(
                f"[SemanticCache] Full ({self.max_entries}), FIFO eviction: "
                f"'{removed['prompt'][:30]}...'"
            )

        emb = self._embed(prompt)
        self._entries.append({
            'prompt':    prompt,
            'embedding': emb,
            'response':  response,
            'hits':      0,
        })
        print(f"[SemanticCache] Stored  (entries~{len(self._entries)})")

    def clear(self) -> None:
        """Removes all cached entries."""
        self._entries.clear()
        self._total_hits = 0
        print("[SemanticCache] Cache cleared.")

    def stats(self) -> dict:
        """Returns cache statistics (exposed via /api/model-info)."""
        return {
            'cache_entries':    len(self._entries),
            'cache_hits_total': self._total_hits,
            'cache_threshold':  self.threshold,
            'cache_max_size':   self.max_entries,
            'storage':          'RAM',
            'eviction':         'FIFO',
        }
