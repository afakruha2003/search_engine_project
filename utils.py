"""
utils.py
Bonus features:
  • Wildcard query processing (prefix/suffix matching)
  • Spelling correction (edit distance)
  • Query expansion (synonym-based)
  • Document snippets in search results
  • Evaluation metrics: Precision, Recall, F1-score
"""

import re
from collections import Counter


# ─────────────────────────────────────────────
#  1. Wildcard Query Processing
# ─────────────────────────────────────────────

def wildcard_search(pattern: str, vocabulary: list[str]) -> list[str]:
    """
    Find all vocabulary terms matching a wildcard pattern.
    Supports:
      - Prefix wildcard:  'comp*'  → all terms starting with 'comp'
      - Suffix wildcard:  '*tion'  → all terms ending with 'tion'
      - Infix wildcard:   'com*er' → all terms matching 'com...er'

    Parameters
    ----------
    pattern    : wildcard string (use * as wildcard character)
    vocabulary : list of preprocessed vocabulary terms
    """
    pattern = pattern.lower().strip()
    # Convert wildcard pattern to regex
    regex_pattern = '^' + re.escape(pattern).replace(r'\*', '.*') + '$'
    compiled = re.compile(regex_pattern)
    return [term for term in vocabulary if compiled.match(term)]


# ─────────────────────────────────────────────
#  2. Spelling Correction (Edit Distance)
# ─────────────────────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    Allowed operations: insert, delete, substitute (cost 1 each).
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # delete
                dp[i][j - 1] + 1,       # insert
                dp[i - 1][j - 1] + cost # substitute
            )
    return dp[m][n]


def suggest_corrections(
    term: str,
    vocabulary: list[str],
    max_distance: int = 2,
    top_k: int = 5
) -> list[dict]:
    """
    Suggest vocabulary terms close to the misspelled term.

    Returns list of {'suggestion': term, 'distance': int} sorted by distance.
    """
    term = term.lower()
    candidates = []
    for word in vocabulary:
        dist = edit_distance(term, word)
        if dist <= max_distance:
            candidates.append({'suggestion': word, 'distance': dist})
    candidates.sort(key=lambda x: x['distance'])
    return candidates[:top_k]


# ─────────────────────────────────────────────
#  3. Query Expansion (Simple Synonym Map)
# ─────────────────────────────────────────────

# A curated synonym map for common query terms
SYNONYM_MAP = {
    'ai': ['artificial', 'intelligence', 'machine', 'learning'],
    'ml': ['machine', 'learning', 'algorithm'],
    'climate': ['global', 'warming', 'environment', 'carbon'],
    'energy': ['power', 'electricity', 'renewable', 'solar', 'wind'],
    'medicine': ['health', 'medical', 'disease', 'treatment'],
    'doctor': ['physician', 'medical', 'healthcare'],
    'computer': ['computing', 'digital', 'electronic', 'processor'],
    'internet': ['web', 'network', 'online', 'digital'],
    'robot': ['automation', 'robotic', 'autonomous', 'machine'],
    'gene': ['genetic', 'dna', 'genomics', 'biology'],
    'economy': ['economic', 'finance', 'market', 'trade'],
    'space': ['astronomy', 'cosmos', 'universe', 'galaxy'],
    'data': ['information', 'dataset', 'analytics'],
    'security': ['cybersecurity', 'protection', 'encryption', 'privacy'],
    'food': ['nutrition', 'diet', 'eating', 'nutrient'],
    'school': ['education', 'learning', 'university', 'academic'],
    'city': ['urban', 'metropolitan', 'municipal'],
    'ocean': ['marine', 'sea', 'water', 'aquatic'],
    'blockchain': ['cryptocurrency', 'bitcoin', 'decentralized', 'ledger'],
    'social': ['media', 'network', 'platform', 'community'],
}


def expand_query(tokens: list[str], preprocessor=None) -> list[str]:
    """
    Expand query tokens with synonyms from the SYNONYM_MAP.
    Returns original tokens + expansion terms (deduplicated).

    Parameters
    ----------
    tokens      : preprocessed query tokens
    preprocessor: optional function to preprocess expansion terms
    """
    expanded = list(tokens)
    seen = set(tokens)

    for token in tokens:
        synonyms = SYNONYM_MAP.get(token, [])
        for syn in synonyms:
            if preprocessor:
                processed = preprocessor(syn)
                for p in processed:
                    if p not in seen:
                        expanded.append(p)
                        seen.add(p)
            else:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

    return expanded


# ─────────────────────────────────────────────
#  4. Document Snippets
# ─────────────────────────────────────────────

def generate_snippet(
    raw_text: str,
    query_terms: list[str],
    snippet_length: int = 200
) -> str:
    """
    Extract a relevant snippet from a document that contains query terms.
    Finds the sentence/window with the most query term matches.

    Parameters
    ----------
    raw_text      : original (unprocessed) document text
    query_terms   : list of query terms to highlight context around
    snippet_length: approximate character length of snippet
    """
    if not raw_text or not query_terms:
        return raw_text[:snippet_length] + '...'

    text_lower = raw_text.lower()
    best_pos = 0
    best_count = 0

    # Find window with most query term hits
    window = snippet_length // 2
    for term in query_terms:
        pos = text_lower.find(term)
        while pos != -1:
            start = max(0, pos - window)
            end = min(len(raw_text), pos + window)
            window_text = text_lower[start:end]
            count = sum(window_text.count(t) for t in query_terms)
            if count > best_count:
                best_count = count
                best_pos = start
            pos = text_lower.find(term, pos + 1)

    # Extract snippet around best position
    start = max(0, best_pos)
    end = min(len(raw_text), start + snippet_length)
    snippet = raw_text[start:end].strip()

    # Clean snippet edges
    if start > 0:
        snippet = '...' + snippet
    if end < len(raw_text):
        snippet = snippet + '...'

    # Bold query terms (for HTML rendering)
    for term in query_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        snippet = pattern.sub(f'**{term}**', snippet)

    return snippet


# ─────────────────────────────────────────────
#  5. Evaluation Metrics
# ─────────────────────────────────────────────

def compute_metrics(
    retrieved: list[str],
    relevant: list[str]
) -> dict:
    """
    Compute Precision, Recall, and F1-score.

    Parameters
    ----------
    retrieved : list of doc IDs returned by the system
    relevant  : list of truly relevant doc IDs (ground truth)

    Returns
    -------
    {'precision': float, 'recall': float, 'f1': float,
     'true_positives': int, 'retrieved': int, 'relevant': int}
    """
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    tp = len(retrieved_set & relevant_set)
    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'true_positives': tp,
        'num_retrieved': len(retrieved_set),
        'num_relevant': len(relevant_set)
    }


def average_precision(ranked_results: list[str], relevant: list[str]) -> float:
    """
    Compute Average Precision (AP) for ranked results.
    Used for MAP (Mean Average Precision) evaluation.
    """
    relevant_set = set(relevant)
    hits = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(ranked_results, start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / i

    return round(precision_sum / len(relevant_set), 4) if relevant_set else 0.0
