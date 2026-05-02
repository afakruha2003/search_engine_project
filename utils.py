"""
utils.py
Bonus features:
  • Wildcard query processing (prefix/suffix matching)
  • Spelling correction (edit distance)
  • Query expansion (synonym-based)
  • Document snippets in search results
  • Evaluation metrics: Precision, Recall, F1-score
"""

import re #regular expressions  for regex in wildcard search 




def wildcard_search(pattern: str, vocabulary: list[str]) -> list[str]:
    """Find all vocabulary terms matching a wildcard pattern."""
    pattern = pattern.lower().strip()
    # Convert wildcard pattern to regex
    regex_pattern = '^' + re.escape(pattern).replace(r'\*', '.*') + '$'  # match entire term 
    compiled = re.compile(regex_pattern) 
    return [term for term in vocabulary if compiled.match(term)]




def edit_distance(s1: str, s2: str) -> int:

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)] # create a (m+1) x (n+1) matrix to store edit distances


# fill the first row and column of the matrix with base cases: transforming an empty string to the other string requires insertions or deletions equal to the length of the other string
    for i in range(m + 1): 
        dp[i][0] = i   
    for j in range(n + 1):
        dp[0][j] = j

# fill the rest of the matrix using the recursive relation: if characters match, no cost; otherwise, consider the cost of delete, insert, or substitute
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # delete
                dp[i][j - 1] + 1,       # insert
                dp[i - 1][j - 1] + cost # substitute
            )
    return dp[m][n]

"""
        c  u  t
    [0, 0, 0, 0]
c   [0, 0, 0, 0]
a   [0, 0, 0, 0]
t   [0, 0, 0, 0]

        c  u  t
    [0, 1, 2, 3]
c   [1, 0, 0, 0]
a   [2, 0, 0, 0]
t   [3, 0, 0, 0]

for example, when we compare
s1[0]='c' with s2[0]='c'
dp[1][1] = min(
    dp[0][1] + 1 = 1+1 = 2,   # delete
    dp[1][0] + 1 = 1+1 = 2,   # insert
    dp[0][0] + 0 = 0+0 = 0    # subsitiute
) = 0

          c  u  t
      [0, 1, 2, 3]
  c   [1, 0, 1, 2]
  a   [2, 1, 1, 2]
  t   [3, 2, 2, 1]  ← = 1
"""
# It uses dynamic programming to fill a matrix dp where dp[i][j] represents the minimum edit distance between the first i characters of s1 and the first j characters of s2. The function returns the value in the bottom-right cell of the matrix, which is the edit distance between the full strings.




def suggest_corrections(term: str,vocabulary: list[str],max_distance: int = 2, top_k: int = 5) -> list[dict]:

    term = term.lower()
    candidates = []
    for word in vocabulary:
        dist = edit_distance(term, word)
        if dist <= max_distance:
            candidates.append({'suggestion': word, 'distance': dist})
    candidates.sort(key=lambda x: x['distance'])
    return candidates[:top_k]





#  3. Query Expansion (Simple Synonym Map)

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

    expanded = list(tokens)
    seen = set(tokens) # to avoid duplicates

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
#for each token, we look up synonyms in the SYNONYM_MAP. If a preprocessor is provided, we process each synonym and add the processed terms to the expanded list, ensuring no duplicates. If no preprocessor is given, we add the raw synonyms directly.


#  5. Evaluation Metrics
def compute_metrics( retrieved: list[str], relevant: list[str]) -> dict:

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
    relevant_set = set(relevant)
    hits = 0 # count of relevant documents found so far
    precision_sum = 0.0 # sum of precision values at ranks where relevant docs are found

    for i, doc_id in enumerate(ranked_results, start=1): 
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / i

    return round(precision_sum / len(relevant_set), 4) if relevant_set else 0.0
# It iterates through the ranked results, counting how many relevant documents have been found (hits) and summing the precision at each rank where a relevant document is retrieved. Finally, it divides this sum by the total number of relevant documents to get the average precision score.