import pandas as pd
import numpy as np

class IncidenceMatrix:
    def __init__(self):
        self.matrix: pd.DataFrame = None   # DataFrame[term x doc]
        self.vocabulary: list[str] = []
        self.doc_ids: list[str] = []

    def build(self, processed_docs: dict[str, list[str]]) -> pd.DataFrame:
        self.doc_ids = sorted(processed_docs.keys())

        # Collect vocabulary
        vocab_set = set()
        for tokens in processed_docs.values():
            vocab_set.update(tokens)
        self.vocabulary = sorted(vocab_set)

        # Fill matrix
        data = {}
        for doc_id in self.doc_ids:
            token_set = set(processed_docs[doc_id])
            data[doc_id] = [1 if term in token_set else 0
                            for term in self.vocabulary]

        self.matrix = pd.DataFrame(data, index=self.vocabulary)
        return self.matrix

    
    
    #  Query

    def _get_vector(self, term: str) -> np.ndarray:
        """Return binary vector for a term (zeros if not in vocab)."""
        if term in self.matrix.index:
            return self.matrix.loc[term].values.astype(int)
        return np.zeros(len(self.doc_ids), dtype=int)


    def query_and(self, terms: list[str]) -> list[str]:
        if not terms:
            return []
        result = self._get_vector(terms[0])
        for term in terms[1:]:
            result = result & self._get_vector(term)
        return [self.doc_ids[i] for i, v in enumerate(result) if v == 1]

    def query_or(self, terms: list[str]) -> list[str]:
        if not terms:
            return []
        result = self._get_vector(terms[0])
        for term in terms[1:]:
            result = result | self._get_vector(term)
        return [self.doc_ids[i] for i, v in enumerate(result) if v == 1]

    def query_not(self, term: str) -> list[str]:
        vec = self._get_vector(term)
        result = 1 - vec
        return [self.doc_ids[i] for i, v in enumerate(result) if v == 1]

    def boolean_query(self, query: str, preprocessor) -> list[str]:
        query = query.strip() # Remove leading/trailing whitespace
        parts = query.upper().split()

        # NOT query
        if parts[0] == 'NOT':
            raw_term = query.split(' ', 1)[1]
            terms = preprocessor(raw_term)
            return self.query_not(terms[0]) if terms else []

        # Detect operator
        if 'AND' in parts:
            idx = parts.index('AND')
            raw_query_clean = query.replace(' AND ', ' ').replace(' and ', ' ')
        elif 'OR' in parts:
            idx = parts.index('OR')
            raw_query_clean = query.replace(' OR ', ' ').replace(' or ', ' ')
        else:
            idx = None
            raw_query_clean = query

        # Preprocess all terms
        terms = preprocessor(raw_query_clean)

        if 'AND' in parts:
            return self.query_and(terms)
        elif 'OR' in parts:
            return self.query_or(terms)
        else:
            return self.query_or(terms)





    #  Display helpers

    def get_sample(self, n_terms: int , n_docs: int ) -> pd.DataFrame:
        if self.matrix is None:
            return pd.DataFrame()
        return self.matrix.iloc[:n_terms, :n_docs]

    def get_stats(self) -> dict:
        return {
            'vocabulary_size': len(self.vocabulary),
            'num_documents': len(self.doc_ids),
            'matrix_shape': f"{len(self.vocabulary)} × {len(self.doc_ids)}",
            'total_cells': len(self.vocabulary) * len(self.doc_ids),
            'non_zero_cells': int(self.matrix.values.sum()) if self.matrix is not None else 0,
            'sparsity': (
                round(1 - self.matrix.values.sum() /
                      (len(self.vocabulary) * len(self.doc_ids)), 4)
                if self.matrix is not None and len(self.vocabulary) > 0 else 0
            )
        }


