"""
Sentence embedding wrapper using sentence-transformers.
all-MiniLM-L6-v2 maps text to 384-dimensional vectors optimised for semantic similarity.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        print(f"Loading model '{model_name}' on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Model ready — embedding dimension: {self.dim}")

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Return L2-normalised embeddings of shape (N, dim).
        Normalisation makes cosine similarity equivalent to dot product,
        speeding up large-scale retrieval .
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # unit vectors → dot product == cosine sim
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_contexts(self, contexts: List[str], desc: str = "Embedding contexts") -> np.ndarray:
        return self.embed(contexts, show_progress=True)

    def embed_candidates_batched(
        self,
        candidates_per_example: List[List[str]],
        desc: str = "Embedding candidates",
    ) -> List[np.ndarray]:
        """
        Embed candidates for each example separately, returning a list
        of arrays shaped (num_candidates, dim).
        """
        all_candidates = [c for pool in candidates_per_example for c in pool]
        flat_embeddings = self.embed(all_candidates, show_progress=True)

        result = []
        offset = 0
        for pool in candidates_per_example:
            n = len(pool)
            result.append(flat_embeddings[offset: offset + n])
            offset += n
        return result
