"""
inverted_index.py
Build an inverted index and support Boolean AND / OR / NOT queries
with posting list merging.
"""

from collections import defaultdict


class InvertedIndex:
    """
    Inverted index: maps each term to its posting list.
    Posting list = sorted list of doc IDs containing the term.
    Also stores document frequency (df) per term.
    """

    def __init__(self):
        # {term: {'df': int, 'postings': [doc_id, ...]}}
        self.index: dict = {}
        self.doc_ids: list[str] = []

    # ─────────────────────────────────────────
    #  Build
    # ─────────────────────────────────────────

    def build(self, processed_docs: dict[str, list[str]]):
        """
        Build inverted index from preprocessed documents.

        Parameters
        ----------
        processed_docs : {doc_id: [tokens]}
        """
        self.doc_ids = sorted(processed_docs.keys())
        raw_index: dict[str, set] = defaultdict(set)

        for doc_id, tokens in processed_docs.items():
            for token in tokens:
                raw_index[token].add(doc_id)

        self.index = {
            term: {
                'df': len(postings),
                'postings': sorted(postings)
            }
            for term, postings in sorted(raw_index.items())
        }

    # ─────────────────────────────────────────
    #  Posting list helpers
    # ─────────────────────────────────────────

    def get_postings(self, term: str) -> list[str]:
        """Return posting list for a term."""
        if term in self.index:
            return self.index[term]['postings']
        return []

    def _intersect(self, p1: list[str], p2: list[str]) -> list[str]:
        """Merge two posting lists with AND (linear merge)."""
        result, i, j = [], 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i])
                i += 1; j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        return result

    def _union(self, p1: list[str], p2: list[str]) -> list[str]:
        """Merge two posting lists with OR."""
        result, i, j = [], 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i])
                i += 1; j += 1
            elif p1[i] < p2[j]:
                result.append(p1[i]); i += 1
            else:
                result.append(p2[j]); j += 1
        result.extend(p1[i:])
        result.extend(p2[j:])
        return result

    def _complement(self, postings: list[str]) -> list[str]:
        """NOT: all doc IDs not in posting list."""
        posting_set = set(postings)
        return [d for d in self.doc_ids if d not in posting_set]

    # ─────────────────────────────────────────
    #  Query
    # ─────────────────────────────────────────

    def boolean_query(self, query: str, preprocessor) -> list[str]:
        """
        Parse and execute Boolean query using posting list merge.
        Supports: term AND term, term OR term, NOT term, single term.
        """
        query = query.strip()
        upper = query.upper()

        if upper.startswith('NOT '):
            raw_term = query[4:]
            terms = preprocessor(raw_term)
            if not terms:
                return []
            return self._complement(self.get_postings(terms[0]))

        if ' AND ' in upper:
            parts = query.upper().split(' AND ')
            term_lists = [preprocessor(p.strip()) for p in query.split(' AND ') + query.split(' and ')]
            # Rebuild clean split
            raw_parts = query.replace(' AND ', ' AND ').split(' AND ')
            if len(raw_parts) < 2:
                raw_parts = query.split(' and ')

            processed_parts = [preprocessor(p.strip()) for p in raw_parts]
            result = self.get_postings(processed_parts[0][0]) if processed_parts[0] else []
            for part_tokens in processed_parts[1:]:
                if part_tokens:
                    result = self._intersect(result, self.get_postings(part_tokens[0]))
            return result

        if ' OR ' in upper:
            raw_parts = query.replace(' OR ', ' OR ').split(' OR ')
            if len(raw_parts) < 2:
                raw_parts = query.split(' or ')
            processed_parts = [preprocessor(p.strip()) for p in raw_parts]
            result = []
            for part_tokens in processed_parts:
                if part_tokens:
                    result = self._union(result, self.get_postings(part_tokens[0]))
            return result

        # Single term
        terms = preprocessor(query)
        if not terms:
            return []
        return self.get_postings(terms[0])

    def search(self, term: str) -> dict:
        """Return full entry for a term."""
        if term in self.index:
            return {'term': term, **self.index[term]}
        return {'term': term, 'df': 0, 'postings': []}

    # ─────────────────────────────────────────
    #  Display helpers
    # ─────────────────────────────────────────

    def get_sample_entries(self, n: int ) -> list[dict]:
        """Return first n entries for display."""
        entries = []
        for i, (term, data) in enumerate(self.index.items()):
            if i >= n:
                break
            entries.append({
                'term': term,
                'df': data['df'],
                'postings': ', '.join(data['postings'])
            })
        return entries

    def get_stats(self) -> dict:
        return {
            'total_terms': len(self.index),
            'total_documents': len(self.doc_ids),
            'avg_postings_per_term': round(
                sum(v['df'] for v in self.index.values()) / len(self.index), 2
            ) if self.index else 0
        }
