"""
positional_index.py
Positional index supporting exact phrase queries.
Stores {term: {doc_id: [positions]}} for consecutive position matching.
"""

from collections import defaultdict


class PositionalIndex:
    """
    Positional index: stores term positions within each document.
    Enables phrase query matching (e.g., "machine learning").
    """

    def __init__(self):
        # {term: {doc_id: [pos1, pos2, ...]}}
        self.index: dict[str, dict[str, list[int]]] = {}
        self.doc_ids: list[str] = []

    # ─────────────────────────────────────────
    #  Build
    # ─────────────────────────────────────────

    def build(self, processed_docs: dict[str, list[str]]):
        """
        Build positional index from preprocessed documents.

        Parameters
        ----------
        processed_docs : {doc_id: [tokens_in_order]}
        """
        self.doc_ids = sorted(processed_docs.keys())
        raw_index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

        for doc_id, tokens in processed_docs.items():
            for position, token in enumerate(tokens):
                raw_index[token][doc_id].append(position)

        # Convert to plain dicts
        self.index = {
            term: {doc_id: positions for doc_id, positions in doc_map.items()}
            for term, doc_map in sorted(raw_index.items())
        }

    # ─────────────────────────────────────────
    #  Phrase query
    # ─────────────────────────────────────────

    def phrase_query(self, phrase_tokens: list[str]) -> list[str]:
        """
        Find documents where all phrase tokens appear consecutively
        in the correct order.

        Parameters
        ----------
        phrase_tokens : already preprocessed list of terms in the phrase
        """
        if not phrase_tokens:
            return []

        if len(phrase_tokens) == 1:
            term = phrase_tokens[0]
            if term in self.index:
                return sorted(self.index[term].keys())
            return []

        # Start with candidate docs from first term
        first_term = phrase_tokens[0]
        if first_term not in self.index:
            return []

        candidate_docs = set(self.index[first_term].keys())

        # Intersect candidates across all terms
        for term in phrase_tokens[1:]:
            if term not in self.index:
                return []
            candidate_docs &= set(self.index[term].keys())

        # For each candidate doc, check consecutive positions
        matching_docs = []
        for doc_id in sorted(candidate_docs):
            if self._positions_match(doc_id, phrase_tokens):
                matching_docs.append(doc_id)

        return matching_docs

    def _positions_match(self, doc_id: str, phrase_tokens: list[str]) -> bool:
        """
        Check whether phrase_tokens appear consecutively in doc_id.
        For each starting position of the first term, verify that
        term[k] appears at position(start + k).
        """
        first_positions = self.index[phrase_tokens[0]].get(doc_id, [])

        for start_pos in first_positions:
            match = True
            for k, term in enumerate(phrase_tokens[1:], start=1):
                expected_pos = start_pos + k
                term_positions = self.index[term].get(doc_id, [])
                if expected_pos not in term_positions:
                    match = False
                    break
            if match:
                return True

        return False

    # ─────────────────────────────────────────
    #  Single-term lookup
    # ─────────────────────────────────────────

    def lookup(self, term: str) -> dict:
        """Return positional data for a single term."""
        if term in self.index:
            return {
                'term': term,
                'df': len(self.index[term]),
                'postings': {
                    doc_id: positions
                    for doc_id, positions in self.index[term].items()
                }
            }
        return {'term': term, 'df': 0, 'postings': {}}

    # ─────────────────────────────────────────
    #  Display helpers
    # ─────────────────────────────────────────

    def get_sample_entries(self, n: int = 15) -> list[dict]:
        """Return first n entries for display."""
        entries = []
        for i, (term, doc_map) in enumerate(self.index.items()):
            if i >= n:
                break
            postings_str = '; '.join(
                f"{doc_id}: [{', '.join(map(str, pos[:5]))}{'...' if len(pos) > 5 else ''}]"
                for doc_id, pos in doc_map.items()
            )
            entries.append({
                'term': term,
                'df': len(doc_map),
                'positions': postings_str
            })
        return entries

    def get_stats(self) -> dict:
        total_positions = sum(
            len(pos)
            for doc_map in self.index.values()
            for pos in doc_map.values()
        )
        return {
            'total_terms': len(self.index),
            'total_documents': len(self.doc_ids),
            'total_positions_stored': total_positions
        }
