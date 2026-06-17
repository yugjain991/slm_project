# knowledge_base.py
# RAG-powered semantic search Q&A system using sentence-transformers and FAISS.
# Provides embedding-based retrieval for company knowledge queries.

import os
import pickle

# Force single-threading for BLAS/OpenMP libraries to avoid OpenBLAS memory allocation failure
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Resolve paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class KnowledgeBase:
    """
    Builds and queries a FAISS-backed semantic search index over
    Q&A pairs and factual statements about BrightTech Solutions.
    Uses cosine similarity (inner product on L2-normalized vectors).
    """

    def __init__(self, index_path="knowledge.index", data_path="knowledge_data.pkl"):
        """
        Loads the FAISS index and Q&A metadata from disk if they exist.
        Args:
            index_path: filename for the FAISS index file.
            data_path:  filename for the pickled metadata (questions + answers).
        """
        self.index_path = os.path.join(BASE_DIR, index_path)
        self.data_path = os.path.join(BASE_DIR, data_path)
        self.model = None  # Lazy-loaded SentenceTransformer
        self.index = None
        self.metadata = []  # List of {'question': ..., 'answer': ...} dicts

        # Load existing index + metadata from disk if available
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.data_path, "rb") as f:
                self.metadata = pickle.load(f)

    def _load_model(self):
        """Lazy-loads the SentenceTransformer model on first use."""
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build(self, qa_pairs, facts):
        """
        Builds the FAISS index from Q&A pairs and standalone facts.

        Args:
            qa_pairs: list of (question, answer) tuples.
            facts:    list of fact strings (stored as both question and answer).
        """
        self._load_model()

        # Combine Q&A pairs and facts into a unified metadata list
        self.metadata = []
        texts_to_embed = []

        for question, answer in qa_pairs:
            self.metadata.append({"question": question, "answer": answer})
            texts_to_embed.append(question)

        # Facts are indexed by their own text (question == answer)
        for fact in facts:
            self.metadata.append({"question": fact, "answer": fact})
            texts_to_embed.append(fact)

        # Generate embeddings and normalize for cosine similarity
        embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        # Build FAISS IndexFlatIP (inner product ≈ cosine on normalized vecs)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Persist to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def search(self, query, top_k=3, threshold=0.5):
        """
        Embeds the query, searches the FAISS index, and returns results
        above the similarity threshold.

        Args:
            query:     the search query string.
            top_k:     maximum number of results to return.
            threshold: minimum cosine similarity score to include a result.

        Returns:
            List of (answer, score) tuples, sorted by descending score.
            Returns empty list if no index is loaded.
        """
        if self.index is None or not self.metadata:
            return []

        self._load_model()

        # Encode and normalize the query vector
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)

        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for unfilled slots
            if score >= threshold:
                results.append((self.metadata[idx]["answer"], float(score)))

        return results

    def get_best_answer(self, query, threshold=0.6):
        """
        Convenience method: returns the top-1 answer if its score
        meets the threshold, otherwise None.

        Args:
            query:     the search query string.
            threshold: minimum cosine similarity score.

        Returns:
            The best answer string, or None.
        """
        results = self.search(query, top_k=1, threshold=threshold)
        if results:
            return results[0][0]
        return None
