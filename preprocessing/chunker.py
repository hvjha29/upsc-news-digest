"""Chunking utilities.

This module provides `chunk_text()` which supports token-accurate chunking
when `tiktoken` is available. If `tiktoken` is not installed, it falls back
to a character-approximation chunker (the previous behaviour).

API
---
- chunk_text(text, max_tokens=500, overlap=50, tokenizer=None, chars_per_token=4)
	Return overlapping chunks. If `tokenizer` is provided it should be a
	callable with `encode(text) -> List[int]` and `decode(List[int]) -> str`.
"""

from typing import List, Optional, Callable
import warnings


def _char_chunk_text(
	text: str, max_tokens: int = 500, overlap: int = 50, chars_per_token: int = 4
) -> List[str]:
	"""Fallback character-approximate chunking.

	Kept for environments without a tokenizer. Behaviour mirrors earlier
	implementation: prefer splitting on whitespace, apply overlap, return
	non-empty chunks.
	"""
	if not text:
		return []

	max_chars = max(1, int(max_tokens * chars_per_token))
	overlap_chars = max(0, int(overlap * chars_per_token))

	chunks: List[str] = []
	text_len = len(text)
	start = 0

	while start < text_len:
		end = start + max_chars
		if end >= text_len:
			chunks.append(text[start:].strip())
			break

		window = text[start:end]
		last_space = window.rfind(" ")
		last_newline = window.rfind("\n")
		split_at = max(last_space, last_newline)
		if split_at <= 0:
			split_at = max_chars

		chunks.append(text[start : start + split_at].strip())

		advance = split_at - overlap_chars
		if advance <= 0:
			advance = max(1, max_chars - overlap_chars)
		start = start + advance

	return [c for c in chunks if c]


def chunk_text(
	text: str,
	max_tokens: int = 500,
	overlap: int = 50,
	tokenizer: Optional[Callable] = None,
	chars_per_token: int = 4,
) -> List[str]:
	"""Chunk text into overlapping segments.

	Args:
		text: Input text.
		max_tokens: Maximum tokens per chunk (approx or exact depending on tokenizer).
		overlap: Overlap in tokens between chunks.
		tokenizer: Optional tokenizer object or callable that supports
			`encode(text) -> List[int]` and `decode(List[int]) -> str`. If
			omitted the function will attempt to use `tiktoken`.
		chars_per_token: When tokenizer not available, use this to approx tokens->chars.

	Returns:
		List of overlapping text chunks.
	"""
	if not text:
		return []

	# If a tokenizer is provided, use it. Otherwise try to import tiktoken.
	enc = None
	if tokenizer is not None:
		enc = tokenizer
	else:
		try:
			import tiktoken

			enc = tiktoken.get_encoding("cl100k_base")
		except Exception:
			enc = None

	if enc is None:
		warnings.warn(
			"Token tokenizer not available; falling back to character approximation. "
			"Install 'tiktoken' for token-accurate chunking.",
			UserWarning,
		)
		return _char_chunk_text(text, max_tokens=max_tokens, overlap=overlap, chars_per_token=chars_per_token)

	# Token-aware chunking
	# enc should provide .encode and .decode
	try:
		tokens = enc.encode(text)
	except Exception:
		# some tokenizer objects use encode_ordinary or different API
		# attempt to call as a callable
		tokens = enc(text)

	max_t = max(1, int(max_tokens))
	overlap_t = max(0, int(overlap))

	chunks: List[str] = []
	i = 0
	n = len(tokens)
	while i < n:
		j = i + max_t
		if j >= n:
			token_slice = tokens[i:]
			try:
				chunk = enc.decode(token_slice)
			except Exception:
				# if decode not present, join via space (best-effort)
				chunk = "".join([str(t) for t in token_slice])
			chunks.append(chunk.strip())
			break

		token_slice = tokens[i:j]
		try:
			chunk = enc.decode(token_slice)
		except Exception:
			chunk = "".join([str(t) for t in token_slice])

		# If decoded chunk is empty for some reason, fall back to char chunk
		if not chunk.strip():
			return _char_chunk_text(text, max_tokens=max_tokens, overlap=overlap, chars_per_token=chars_per_token)

		chunks.append(chunk.strip())

		# advance pointer by max_t - overlap_t
		advance = max_t - overlap_t
		if advance <= 0:
			advance = 1
		i += advance

	return [c for c in chunks if c]


if __name__ == "__main__":
	# Smoke tests for both modes
	s = " ".join(["word"] * 1000)
	print("Char-approx chunks:")
	ch = _char_chunk_text(s, max_tokens=50, overlap=10)
	print(len(ch), [len(x) for x in ch[:3]])

	try:
		import tiktoken

		print("Token-accurate chunks (tiktoken):")
		ch2 = chunk_text(s, max_tokens=50, overlap=10)
		print(len(ch2), [len(x) for x in ch2[:3]])
	except Exception:
		print("tiktoken not available; skipped token test")

