"""
ranking.py
TF-IDF weighting and Cosine Similarity for ranked retrieval.
All formulas implemented from scratch using math library only.
"""

import math
from collections import Counter
class TFIDFRanker:
    """
    Ranked retrieval using TF-IDF weighting and Cosine Similarity.

    Formulas used:
        tf(t, d)    = count of t in d / total tokens in d
        idf(t)      = log(N / df(t))          [base-10 log]
        tfidf(t, d) = tf(t, d) × idf(t)
        cosine(q,d) = (q · d) / (|q| × |d|)
    """

    def __init__(self):
        self.tfidf_matrix: dict[str, dict[str, float]] = {}  # {doc_id: {term: tfidf}}
        self.idf: dict[str, float] = {}
        self.doc_ids: list[str] = []
        self.vocabulary: list[str] = []
        self.N: int = 0

    # ─────────────────────────────────────────
    #  Build
    # ─────────────────────────────────────────

    def build(self, processed_docs: dict[str, list[str]]):
        """
        Compute TF-IDF matrix for all documents.

        Parameters
        ----------
        processed_docs : {doc_id: [tokens]}
        """
        self.doc_ids = sorted(processed_docs.keys())
        self.N = len(self.doc_ids)

        # Count term frequencies per document
        tf_counts: dict[str, Counter] = {
            doc_id: Counter(tokens)
            for doc_id, tokens in processed_docs.items()
        }

        # Build vocabulary
        vocab_set = set()
        for counter in tf_counts.values():
            vocab_set.update(counter.keys())
        self.vocabulary = sorted(vocab_set)

        # Compute DF (document frequency)
        df: dict[str, int] = {}
        for term in self.vocabulary:
            df[term] = sum(
                1 for counter in tf_counts.values() if term in counter
            )

        # Compute IDF
        self.idf = {}
        for term in self.vocabulary:
            # Avoid division by zero
            self.idf[term] = math.log10(self.N / df[term]) if df[term] > 0 else 0.0

        # Compute TF-IDF for each document
        self.tfidf_matrix = {}
        for doc_id, counter in tf_counts.items():
            total_tokens = sum(counter.values())
            doc_tfidf = {}
            for term, count in counter.items():
                tf = count / total_tokens if total_tokens > 0 else 0
                doc_tfidf[term] = tf * self.idf.get(term, 0)
            self.tfidf_matrix[doc_id] = doc_tfidf

    # ─────────────────────────────────────────
    #  Query vector
    # ─────────────────────────────────────────

    def _query_vector(self, query_tokens: list[str]) -> dict[str, float]:
        """
        Convert query tokens into a TF-IDF weighted vector.
        Uses same IDF values as the document collection.
        """
        counter = Counter(query_tokens)
        total = sum(counter.values())
        q_vec = {}
        for term, count in counter.items():
            tf = count / total if total > 0 else 0
            idf = self.idf.get(term, 0)
            q_vec[term] = tf * idf
        return q_vec

    # ─────────────────────────────────────────
    #  Cosine Similarity
    # ─────────────────────────────────────────

    @staticmethod
    def _dot_product(vec_a: dict, vec_b: dict) -> float:
        """Compute dot product of two sparse vectors."""
        common = set(vec_a.keys()) & set(vec_b.keys())
        return sum(vec_a[t] * vec_b[t] for t in common)

    @staticmethod
    def _magnitude(vec: dict) -> float:
        """Compute L2 magnitude of a vector."""
        return math.sqrt(sum(v ** 2 for v in vec.values()))

    def cosine_similarity(self, vec_a: dict, vec_b: dict) -> float:
        """
        Compute cosine similarity between two TF-IDF vectors.
        cosine = (A · B) / (|A| × |B|)
        """
        mag_a = self._magnitude(vec_a)
        mag_b = self._magnitude(vec_b)
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return self._dot_product(vec_a, vec_b) / (mag_a * mag_b)

    # ─────────────────────────────────────────
    #  Ranked retrieval
    # ─────────────────────────────────────────

    def rank(self, query_tokens: list[str], top_k: int = None) -> list[dict]:
        """
        Rank all documents by cosine similarity to the query.

        Parameters
        ----------
        query_tokens : preprocessed query tokens
        top_k        : return top k results (None = all)

        Returns
        -------
        List of {'rank', 'doc_id', 'score'} sorted by score descending.
        """
        if not query_tokens:
            return []

        q_vec = self._query_vector(query_tokens)

        scores = []
        for doc_id, doc_vec in self.tfidf_matrix.items():
            score = self.cosine_similarity(q_vec, doc_vec)
            if score > 0:
                scores.append({'doc_id': doc_id, 'score': round(score, 6)})

        # Sort by score descending
        scores.sort(key=lambda x: x['score'], reverse=True)

        if top_k:
            scores = scores[:top_k]

        # Add rank
        for i, entry in enumerate(scores, start=1):
            entry['rank'] = i

        return scores

    # ─────────────────────────────────────────
    #  Inspect TF-IDF values
    # ─────────────────────────────────────────

    def get_top_terms(self, doc_id: str, n: int = 10) -> list[dict]:
        """Return top n TF-IDF weighted terms for a document."""
        if doc_id not in self.tfidf_matrix:
            return []
        items = sorted(
            self.tfidf_matrix[doc_id].items(),
            key=lambda x: x[1], reverse=True
        )[:n]
        return [{'term': t, 'tfidf': round(v, 6)} for t, v in items]

    def get_stats(self) -> dict:
        return {
            'vocabulary_size': len(self.vocabulary),
            'num_documents': self.N,
            'avg_unique_terms_per_doc': round(
                sum(len(v) for v in self.tfidf_matrix.values()) / self.N, 1
            ) if self.N > 0 else 0
        }
