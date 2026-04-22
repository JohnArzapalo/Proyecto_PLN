"""
Step 4 — Model Classifier

Classifies an incoming query as 'fast' or 'slow' and routes it to the
appropriate model runner.

Current behavior:
  Both 'fast' and 'slow' signals route to Fast Hannah because
  Slow Hannah (Qwen2.5-14B-Instruct) is not yet integrated.

To integrate Slow Hannah in the future:
  classifier = ModelClassifier(fast_runner=..., slow_runner=slow_hannah_runner)
  That is all — no other refactoring is needed.

Classification heuristics (can be replaced by an ML classifier later):
  - Input token count > SLOW_THRESHOLD  →  'slow'
  - Prompt contains a 'complex' keyword →  'slow'
  - Otherwise                           →  'fast'
"""

from __future__ import annotations
from typing import Callable

# Type alias: a runner is any callable that takes token_ids and returns a string
Runner = Callable[[list[int]], str]


class ModelClassifier:
    """
    Heuristic-based query classifier and model router.

    Usage:
        classifier = ModelClassifier(fast_runner=fast_fn, slow_runner=None)

        signal, runner = classifier.route(user_prompt, token_count)
        response = runner(token_ids)
    """

    SLOW_THRESHOLD: int = 80  # input tokens; above this → slow signal

    COMPLEX_KEYWORDS: frozenset[str] = frozenset({
        'explain', 'analyze', 'analyse', 'compare', 'summarize', 'summarise',
        'elaborate', 'describe in detail', 'what is the difference',
        'how does', 'why does', 'write a story', 'write an essay',
        'write a poem', 'in detail', 'step by step', 'give me a detailed',
    })

    def __init__(
        self,
        fast_runner: Runner,
        slow_runner: Runner | None = None,
    ):
        """
        Args:
            fast_runner: callable(token_ids: list[int]) -> str
                         Points to Fast Hannah inference.
            slow_runner: callable(token_ids: list[int]) -> str  |  None
                         Points to Slow Hannah inference (not available yet).
        """
        self.fast_runner = fast_runner
        self.slow_runner = slow_runner  # Register Slow Hannah here when ready

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, prompt: str, token_count: int = 0) -> str:
        """
        Determines query complexity.

        Args:
            prompt:      raw user message (not the full formatted prompt)
            token_count: total input tokens produced by TokenSequenceHandler

        Returns:
            'fast' or 'slow'
        """
        if token_count > self.SLOW_THRESHOLD:
            return 'slow'
        lower = prompt.lower()
        if any(kw in lower for kw in self.COMPLEX_KEYWORDS):
            return 'slow'
        return 'fast'

    def route(self, prompt: str, token_count: int = 0) -> tuple[str, Runner]:
        """
        Classifies the prompt and returns (signal, runner).

        The runner is always fast_runner until slow_runner is registered.
        This means the 'slow' signal is emitted correctly for future use,
        but execution still falls back to Fast Hannah in the meantime.

        Returns:
            (signal, runner)
              signal  → 'fast' | 'slow'
              runner  → callable(token_ids) → str
        """
        signal = self.classify(prompt, token_count)

        if signal == 'slow' and self.slow_runner is not None:
            runner = self.slow_runner
            label  = 'SlowHannah'
        else:
            runner = self.fast_runner
            label  = 'FastHannah'
            if signal == 'slow':
                label += ' (fallback — SlowHannah not integrated yet)'

        print(f"[Classifier] signal={signal} -> {label}")
        return signal, runner
