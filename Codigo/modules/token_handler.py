"""
Step 2 — Token Sequence Handler

Receives the raw conversation history and produces a formatted,
tokenized payload ready for the Semantic Cache and model inference.

Responsibilities:
- Format the prompt using Hannah's [SYS]/[USR]/[ASS] template
- Truncate oldest turns when the total token count exceeds the context budget
- Return a clean dict that downstream modules can consume directly
"""

import sentencepiece as spm
import config


class TokenSequenceHandler:
    """
    Prepares a conversation history for downstream pipeline modules.

    Usage:
        handler = TokenSequenceHandler(tokenizer)
        payload = handler.prepare(conversation)
        # payload['token_ids'] → feed to model
        # payload['user_prompt'] → feed to semantic cache / classifier
    """

    def __init__(self, tokenizer: spm.SentencePieceProcessor):
        self.tokenizer = tokenizer
        # Reserve tokens for the model's own generation budget
        self.context_budget = config.MAX_SEQ_LEN - config.MAX_NEW_TOKENS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, history: list[dict]) -> str:
        """Formats conversation history using Hannah's special-token template."""
        prompt = f"[SYS] {config.SYSTEM_PROMPT} [/SYS]"
        for msg in history:
            if msg['role'] == 'user':
                prompt += f"[USR] {msg['content']} [/USR]"
            else:
                prompt += f"[ASS] {msg['content']} [/ASS]"
        prompt += "[ASS]"
        return prompt

    def _get_user_prompt(self, history: list[dict]) -> str:
        """Returns the most recent user message from history."""
        for msg in reversed(history):
            if msg['role'] == 'user':
                return msg['content']
        return ''

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, history: list[dict]) -> dict:
        """
        Truncates history if needed, builds the formatted prompt, and tokenizes it.

        Args:
            history: list of {'role': 'user'|'assistant', 'content': str}

        Returns:
            {
                'formatted_prompt':  str,        # full [SYS]...[ASS] string
                'token_ids':         list[int],  # tokenized prompt
                'token_count':       int,        # number of input tokens
                'truncated_history': list[dict], # history after truncation
                'user_prompt':       str,        # latest user message (for cache/classifier)
            }
        """
        working = list(history)

        # Trim the oldest turns until the token count fits within the budget
        while len(working) > 1:
            ids = self.tokenizer.Encode(self._build_prompt(working))
            if len(ids) <= self.context_budget:
                break
            working.pop(0)

        prompt    = self._build_prompt(working)
        token_ids = self.tokenizer.Encode(prompt)

        return {
            'formatted_prompt':  prompt,
            'token_ids':         token_ids,
            'token_count':       len(token_ids),
            'truncated_history': working,
            'user_prompt':       self._get_user_prompt(working),
        }

    def inject_rag(self, truncated_history: list[dict], rag_context: str) -> list[int]:
        """
        Rebuilds the token_ids with RAG context injected right after [/SYS].

        Args:
            truncated_history: already-truncated history from prepare()
            rag_context: formatted string from RAGComponent, e.g.
                         "[MEMORY]Hannah has 360M params.[/MEMORY]"
                         Pass '' or "[MEMORY][/MEMORY]" to skip injection.

        Returns:
            list[int] — final token_ids ready for the model.
        """
        skip = not rag_context or rag_context == '[MEMORY][/MEMORY]'

        prompt = f"[SYS] {config.SYSTEM_PROMPT} [/SYS]"
        if not skip:
            prompt += rag_context
        for msg in truncated_history:
            if msg['role'] == 'user':
                prompt += f"[USR] {msg['content']} [/USR]"
            else:
                prompt += f"[ASS] {msg['content']} [/ASS]"
        prompt += "[ASS]"
        return self.tokenizer.Encode(prompt)
