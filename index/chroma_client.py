# index/chroma_client.py
"""Chroma vector database client (modern API).

Uses chromadb.PersistentClient for on-disk persistence with the latest
Chroma API (v0.4+). Migration from old API is handled automatically.
"""
import chromadb
import os

# You can customize persistence_dir if you want on-disk persistence
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")


def get_client(persist: bool = True):
    """Get or create a Chroma client with persistent storage.

    Args:
        persist: If True, use PersistentClient (on-disk storage).
                 If False, use EphemeralClient (in-memory only).

    Returns:
        A Chroma client instance.
    """
    if persist:
        # Modern persistent client (recommended for production)
        client = chromadb.PersistentClient(path=PERSIST_DIR)
    else:
        # Ephemeral client for testing (in-memory only)
        client = chromadb.EphemeralClient()
    return client


def get_or_create_collection(client, name: str, **kwargs):
    """Get an existing collection or create it if it doesn't exist.

    Args:
        client: Chroma client instance.
        name: Name of the collection.
        **kwargs: Additional arguments passed to create_collection (e.g., metadata).

    Returns:
        A Chroma collection instance.
    """
    try:
        col = client.get_collection(name=name)
    except Exception:
        # Collection doesn't exist; create it with optional metadata
        col = client.create_collection(name=name, **kwargs)
    return col
