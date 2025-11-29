import numpy as np
from sentence_transformers import SentenceTransformer


def get_embedder():
    """Load and return the SentenceTransformer embedder model."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text: str, model=None) -> np.ndarray:
    """
    Embed a single text string.
    
    Args:
        text: The text to embed
        model: Optional pre-loaded SentenceTransformer model
        
    Returns:
        NumPy array representing the embedding
    """
    if model is None:
        model = get_embedder()
    return model.encode(text)


def embed_texts(texts: list, model=None) -> np.ndarray:
    """
    Embed multiple texts efficiently.
    
    Args:
        texts: List of text strings to embed
        model: Optional pre-loaded SentenceTransformer model
        
    Returns:
        NumPy array of embeddings (shape: [num_texts, embedding_dim])
    """
    if model is None:
        model = get_embedder()
    return model.encode(texts)