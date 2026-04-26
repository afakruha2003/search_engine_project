"""
preprocessing.py
Document preprocessing pipeline: tokenization, stop-word removal,
stemming, lemmatization, punctuation removal, and case folding.

NOTE: Uses NLTK when available; falls back to built-in implementations
so the module works even without a network connection.
"""

import re
import os

# ── Try NLTK; fall back gracefully ──────────────────────────────────
try:
    import nltk
    for _r in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
             #tokenizer               #lemmatizer 
        try:
            nltk.download(_r, quiet=True)
        except Exception:
            pass
    from nltk.corpus import stopwords as _sw
    from nltk.stem import PorterStemmer as _PS, WordNetLemmatizer as _WNL
    from nltk.tokenize import word_tokenize as _wt
    STOP_WORDS = set(_sw.words('english'))
    _nltk_available = True
except Exception:
    _nltk_available = False

# ── (used when NLTK is unavailable) ─────
_BUILTIN_STOPWORDS = {
    'a','about','above','after','again','against','all','am','an','and',
    'any','are',"aren't",'as','at','be','because','been','before','being',
    'below','between','both','but','by',"can't",'cannot','could',"couldn't",
    'did',"didn't",'do','does',"doesn't",'doing',"don't",'down','during',
    'each','few','for','from','further','get','got','had',"hadn't",'has',
    "hasn't",'have',"haven't",'having','he',"he'd","he'll","he's",'her',
    'here',"here's",'hers','herself','him','himself','his','how',"how's",
    'i',"i'd","i'll","i'm","i've",'if','in','into','is',"isn't",'it',
    "it's",'its','itself',"let's",'me','more','most',"mustn't",'my',
    'myself','no','nor','not','of','off','on','once','only','or','other',
    'ought','our','ours','ourselves','out','over','own','same',"shan't",
    'she',"she'd","she'll","she's",'should',"shouldn't",'so','some','such',
    'than','that',"that's",'the','their','theirs','them','themselves','then',
    'there',"there's",'these','they',"they'd","they'll","they're","they've",
    'this','those','through','to','too','under','until','up','very','was',
    "wasn't",'we',"we'd","we'll","we're","we've",'were',"weren't",'what',
    "what's",'when',"when's",'where',"where's",'which','while','who',
    "who's",'whom','why',"why's",'with',"won't",'would',"wouldn't",'you',
    "you'd","you'll","you're","you've",'your','yours','yourself','yourselves',
    'also','use','used','using','one','two','three','many','much','may',
    'might','well','just','even','still','first','last','new','old','high',
    'low','large','small','big','little','long','short','good','bad','best',
    'see','make','know','come','go','take','give','find','think','look',
    'want','become','include','however','therefore','although','since','while',
}

if not _nltk_available:
    STOP_WORDS = _BUILTIN_STOPWORDS


# ──(used when NLTK is unavailable) ─────────
class _BuiltinStemmer:
    """Minimal Porter Stemmer rules for offline use."""
    _suffixes = [
        ('ational','ate'),('tional','tion'),('enci','ence'),('anci','ance'),
        ('izer','ize'),('ising','ise'),('izing','ize'),('ising','ise'),
        ('ational','ate'),('ation','ate'),('ations','ate'),
        ('nesses','ness'),('ness',''),('ments','ment'),('ment',''),
        ('fulness','ful'),('fulness','ful'),
        ('lessly','less'),('lessly','less'),
        ('ings','ing'),('ing',''),('ingly',''),
        ('edly','ed'),('edly','ed'),('edly',''),('edly',''),
        ('ies','y'),('ied','y'),('ied',''),
        ('sses','ss'),('ses','s'),
        ('ers','er'),('er',''),
        ('ational','ate'),
        ('ies',''),('es',''),
        ('s',''),
    ]

    def stem(self, word: str) -> str:
        if len(word) <= 3:
            return word
        w = word.lower()
        for suffix, replacement in self._suffixes:
            if w.endswith(suffix) and len(w) - len(suffix) >= 2:
                return w[:-len(suffix)] + replacement
        return w


class _BuiltinLemmatizer:
    """Very simple rule-based lemmatizer for offline use."""
    _map = {
        'running':'run','runs':'run','ran':'run',
        'learning':'learn','learns':'learn','learned':'learn',
        'building':'build','builds':'build','built':'build',
        'computing':'compute','computes':'compute','computed':'compute',
        'processing':'process','processes':'process','processed':'process',
        'storing':'store','stores':'store','stored':'store',
        'using':'use','uses':'use',
        'having':'have','has':'have',
        'being':'be','was':'be','were':'be','is':'be','are':'be',
        'doing':'do','does':'do','did':'do',
        'making':'make','makes':'make','made':'make',
        'taking':'take','takes':'take','took':'take',
        'coming':'come','comes':'come','came':'come',
        'going':'go','goes':'go','went':'go',
        'seeing':'see','sees':'see','saw':'see',
        'searching':'search','searches':'search','searched':'search',
        'indexing':'index','indexes':'index','indexed':'index',
        'retrieving':'retrieve','retrieves':'retrieve','retrieved':'retrieve',
        'generating':'generate','generates':'generate','generated':'generate',
        'producing':'produce','produces':'produce','produced':'produce',
        'providing':'provide','provides':'provide','provided':'provide',
        'including':'include','includes':'include','included':'include',
        'allowing':'allow','allows':'allow','allowed':'allow',
        'causing':'cause','causes':'cause','caused':'cause',
        'increasing':'increase','increases':'increase','increased':'increase',
        'reducing':'reduce','reduces':'reduce','reduced':'reduce',
        'developing':'develop','develops':'develop','developed':'develop',
        'diseases':'disease','viruses':'virus','algorithms':'algorithm',
        'technologies':'technology','studies':'study','libraries':'library',
        'companies':'company','countries':'country','cities':'city',
        'systems':'system','networks':'network','models':'model',
        'methods':'method','tools':'tool','data':'data',
        'documents':'document','queries':'query','terms':'term',
        'users':'user','nodes':'node','layers':'layer',
        'neurons':'neuron','features':'feature','vectors':'vector',
        'matrices':'matrix','indices':'index','analyses':'analysis',
    }
    def lemmatize(self, word: str, pos: str = 'n') -> str:
        return self._map.get(word.lower(), word.lower())


if _nltk_available:
    try:
        _stemmer_obj  = _PS()
        _lemma_obj    = _WNL()
        _tokenize_fn  = _wt
    except Exception:
        _nltk_available = False

if not _nltk_available:
    _stemmer_obj  = _BuiltinStemmer()
    _lemma_obj    = _BuiltinLemmatizer()
    _tokenize_fn  = lambda text: re.findall(r'\b[a-z]+\b', text.lower())


stemmer    = _stemmer_obj
lemmatizer = _lemma_obj


# ─────────────────────────────────────────────
#  Core preprocessing steps
# ─────────────────────────────────────────────

def case_fold(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove punctuation and non-alphabetic characters."""
    return re.sub(r'[^a-z\s]', ' ', text)


def tokenize(text: str) -> list[str]:
    """Split text into tokens."""
    return _tokenize_fn(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stop words."""
    return [t for t in tokens if t not in STOP_WORDS]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Porter Stemmer to reduce words to root form."""
    return [_stemmer_obj.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply WordNet Lemmatizer (or built-in fallback) to reduce words to base form."""
    if _nltk_available:
        return [_lemma_obj.lemmatize(t) for t in tokens]
    return [_lemma_obj.lemmatize(t) for t in tokens]


def filter_short_tokens(tokens: list[str], min_len: int = 2) -> list[str]:
    """Remove tokens shorter than min_len characters."""
    return [t for t in tokens if len(t) >= min_len]


# ─────────────────────────────────────────────
#  Main preprocessing pipeline
# ─────────────────────────────────────────────

def preprocess(
    text: str,
    use_stemming: bool = True,
    use_lemmatization: bool = False,
    show_steps: bool = False
) -> list[str]:
    """
    Full preprocessing pipeline.
    Returns a list of processed tokens.

    Parameters
    ----------
    text              : raw document text
    use_stemming      : apply PorterStemmer
    use_lemmatization : apply WordNetLemmatizer (overrides stemming if both True)
    show_steps        : print intermediate steps for demonstration
    """
    steps = {}

    steps['original'] = text[:200] + ('...' if len(text) > 200 else '')

    lowered = case_fold(text)
    steps['case_folded'] = lowered[:200]

    no_punct = remove_punctuation(lowered)
    steps['no_punctuation'] = no_punct[:200]

    tokens = tokenize(no_punct)
    steps['tokenized'] = tokens[:20]

    no_stop = remove_stopwords(tokens)
    steps['stopwords_removed'] = no_stop[:20]

    filtered = filter_short_tokens(no_stop)

    if use_lemmatization:
        final_tokens = lemmatize_tokens(filtered)
        steps['lemmatized'] = final_tokens[:20]
    elif use_stemming:
        final_tokens = stem_tokens(filtered)
        steps['stemmed'] = final_tokens[:20]
    else:
        final_tokens = filtered

    steps['final_tokens'] = final_tokens[:20]

    if show_steps:
        return final_tokens, steps
    return final_tokens


# ─────────────────────────────────────────────
#  Document loading
# ─────────────────────────────────────────────

def load_documents(folder_path: str) -> dict[str, str]:
    """
    Load all .txt documents from a folder.
    Returns {doc_id: raw_text}
    """
    documents = {}
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            doc_id = filename.replace('.txt', '')
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                documents[doc_id] = f.read()

    return documents


def preprocess_all(
    documents: dict[str, str],
    use_stemming: bool = True,
    use_lemmatization: bool = False
) -> dict[str, list[str]]:
    """
    Preprocess all documents.
    Returns {doc_id: [processed_tokens]}
    """
    return {
        doc_id: preprocess(text, use_stemming, use_lemmatization)
        for doc_id, text in documents.items()
    }


# ─────────────────────────────────────────────
#  Comparison helper (Bonus: stemming vs lemmatization)
# ─────────────────────────────────────────────

def compare_stemming_lemmatization(text: str) -> dict:
    """
    Compare stemming vs lemmatization on sample text.
    Returns a dict with both results side by side.
    """
    lowered = case_fold(text)
    no_punct = remove_punctuation(lowered)
    tokens = tokenize(no_punct)
    no_stop = remove_stopwords(tokens)
    filtered = filter_short_tokens(no_stop)

    stemmed = stem_tokens(filtered)
    lemmatized = lemmatize_tokens(filtered)

    comparison = []
    for orig, stem, lemma in zip(filtered, stemmed, lemmatized):
        comparison.append({
            'original': orig,
            'stemmed': stem,
            'lemmatized': lemma,
            'different': stem != lemma
        })

    return {
        'original_tokens': filtered,
        'stemmed': stemmed,
        'lemmatized': lemmatized,
        'comparison_table': comparison
    }
