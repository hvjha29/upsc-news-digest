import os
import sys
import pytest

# Ensure project root is importable when running pytest from the repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import tiktoken
except Exception:  # pragma: no cover - skip when tiktoken not installed
    tiktoken = None

from preprocessing.chunker import chunk_text


@pytest.mark.skipif(tiktoken is None, reason="tiktoken not installed")
def test_token_chunk_overlap():
    """Verify token-accurate chunking produces overlapping token slices.

    This test reproduces the token-slicing logic and ensures the chunks
    returned by `chunk_text` match the token-slice decoding and that
    consecutive slices overlap by the requested number of tokens.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    text = ("hello world " * 300).strip()

    max_tokens = 50
    overlap = 10

    # full token sequence
    tokens = enc.encode(text)
    n = len(tokens)

    # reproduce token-slice logic to build expected chunks and slices
    expected_chunks = []
    expected_token_slices = []
    i = 0
    while i < n:
        j = i + max_tokens
        if j >= n:
            token_slice = tokens[i:]
            expected_token_slices.append(token_slice)
            expected_chunks.append(enc.decode(token_slice).strip())
            break
        token_slice = tokens[i:j]
        expected_token_slices.append(token_slice)
        expected_chunks.append(enc.decode(token_slice).strip())
        advance = max_tokens - overlap
        if advance <= 0:
            advance = 1
        i += advance

    # call chunk_text and compare
    chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap)

    assert len(chunks) == len(expected_chunks)
    for a, b in zip(chunks, expected_chunks):
        assert a.strip() == b.strip()

    # verify token overlap between adjacent token slices
    for prev_slice, cur_slice in zip(expected_token_slices, expected_token_slices[1:]):
        assert prev_slice[-overlap:] == cur_slice[:overlap]
