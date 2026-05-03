import streamlit as st # for web app
import pandas as pd 
import os # for file handling
import time # for measuring query execution time

from preprocessing import (
    load_documents, preprocess_all, preprocess, compare_stemming_lemmatization 
)
from incidence_matrix import IncidenceMatrix
from inverted_index import InvertedIndex
from positional_index import PositionalIndex
from ranking import TFIDFRanker
from utils import (
    wildcard_search, suggest_corrections, expand_query, compute_metrics)

#  Page Config

st.set_page_config(
    page_title="Mini Search Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
    <style>

   
.main-header {
    background-color: #e8f4fd;
    padding: 20px;
    border-radius: 5px;
    margin-bottom: 20px;
    border: 2px solid #90caf9;
}

.main-header h1 {
    font-size: 26px;
    margin: 0;
    color: #0e0e0f;
}

.main-header p {
    margin: 5px 0 0 0;
    color: #555555;
    font-size: 14px;
}

.result-card {
    background: #f9f9f9;
    border: 1px solid #cccccc;
    border-left: 5px solid #42a5f5;
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #333333;
}

.result-card .rank-badge {
    display: inline-block;
    background: #bbdefb;
    color: #0e0e0f;
    border: 1px solid #90caf9;
    border-radius: 3px;
    width: 26px;
    height: 26px;
    text-align: center;
    line-height: 24px;
    font-size: 13px;
    font-weight: bold;
    margin-right: 8px;
}

.metric-box {
    background: #f0f7ff;
    border: 1px solid #90caf9;
    border-radius: 4px;
    padding: 14px;
    text-align: center;
    color: #222222;
}

.stat-number {
    font-size: 28px;
    font-weight: bold;
    color: #0e0e0f;
}

.stat-label {
    font-size: 13px;
    color: #666666;
    margin-top: 4px;
}

.snippet-text {
    font-size: 13px;
    color: #555555;
    margin-top: 6px;
}

.tag {
    display: inline-block;
    background: #e3f2fd;
    color: #0e0e0f;
    border: 1px solid #90caf9;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 12px;
    margin: 2px;
}

.score-bar-container {
    background: #e0e0e0;
    border-radius: 2px;
    height: 6px;
    width: 100%;
    margin-top: 8px;
}

.index-built {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
    padding: 8px 14px;
    border-radius: 4px;
    font-size: 13px;
    margin-bottom: 6px;
}

.warning-box {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 4px;
    padding: 10px 14px;
    color: #7b5800;
}

.info-box {
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 4px;
    padding: 10px 14px;
    color: #1b5e20;
}
 </style>
""", unsafe_allow_html=True)


#  Session State Init
def init_state():
    defaults = {
        'documents': {},
        'processed_docs': {},
        'inc_matrix': None,
        'inv_index': None,
        'pos_index': None,
        'ranker': None,
        'docs_loaded': False,
        'indexes_built': False,
        'use_stemming': True,
        'use_lemma': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


#  Helpers

def get_preprocessor(use_stemming=True, use_lemma=False):
    def preprocessor(text):
        return preprocess(text, use_stemming=use_stemming, use_lemmatization=use_lemma)
    return preprocessor

def doc_title(doc_id: str, raw_docs: dict) -> str:
    text = raw_docs.get(doc_id, '')
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[0] if lines else doc_id



#  Sidebar
with st.sidebar:
    st.markdown("## Ruhe Search Engine")
    st.markdown("---")

    st.markdown("### Document Folder")
    doc_folder = st.text_input("Path", value="documents", label_visibility="collapsed")

    st.markdown("### Preprocessing Options")
    method = st.radio(
        "Normalization Method",
        ["Stemming ", "Lemmatization", "Both (Compare)"],
        index=0,
        label_visibility="collapsed"
    )
    use_stemming = method in ["Stemming ", "Both (Compare)"]
    use_lemma = method in ["Lemmatization", "Both (Compare)"]

    if st.button("Load & Build Indexes", use_container_width=True):
        with st.spinner("Loading documents..."):
            try:
                docs = load_documents(doc_folder)
                st.session_state['documents'] = docs
                st.session_state['docs_loaded'] = True
                st.session_state['use_stemming'] = use_stemming
                st.session_state['use_lemma'] = use_lemma

                proc_docs = preprocess_all(docs, use_stemming, use_lemma)
                st.session_state['processed_docs'] = proc_docs

            except Exception as e:
                st.error(f"Error loading: {e}")
                st.stop()

        with st.spinner("Building indexes..."):
            try:
                # Incidence Matrix
                im = IncidenceMatrix()
                im.build(proc_docs)
                st.session_state['inc_matrix'] = im

                # Inverted Index
                ii = InvertedIndex()
                ii.build(proc_docs)
                st.session_state['inv_index'] = ii

                # Positional Index
                pi = PositionalIndex()
                pi.build(proc_docs)
                st.session_state['pos_index'] = pi

                # TF-IDF Ranker
                ranker = TFIDFRanker()
                ranker.build(proc_docs)
                st.session_state['ranker'] = ranker

                st.session_state['indexes_built'] = True

            except Exception as e:
                st.error(f"Error building indexes: {e}")
                st.stop()

        st.success(f"Successfully indexed {len(docs)} documents.")

    st.markdown("---")
    if st.session_state['docs_loaded']:
        st.markdown(f"<div class='index-built'>Documents loaded: {len(st.session_state['documents'])}</div>", unsafe_allow_html=True)
        if st.session_state['indexes_built']:
            st.markdown("<div class='index-built'>Status: All indexes ready</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Home", "Preprocessing", "Incidence Matrix",
         "Inverted Index", "Positional Index",
         "Ranked Retrieval", "Boolean Search",
         "Bonus Features", "Evaluation"],
        label_visibility="collapsed"
    )


#  Main Header

st.markdown("""
<div class="main-header">
    <h1>Ruhe Search Engine</h1>
    <p>Information Retrieval System — SENG328 Project</p>
</div>
""", unsafe_allow_html=True)



def require_indexes():
    if not st.session_state['indexes_built']:
        st.markdown("""
        <div class="warning-box">
            Please load documents and build indexes first using the sidebar button.
        </div>
        """, unsafe_allow_html=True)
        st.stop()



if page == "Home":
    col1, col2, col3, col4 = st.columns(4)
    n_docs = len(st.session_state['documents'])
    vocab = len(st.session_state['ranker'].vocabulary) if st.session_state['ranker'] else 0
    n_idx = 4 if st.session_state['indexes_built'] else 0

    for col, val, label in zip([col1, col2, col3, col4],[n_docs, vocab, n_idx, "Ready" if st.session_state['indexes_built'] else "Pending"],  ["Documents", "Vocabulary Size", "Indexes Built", "System Status"]):
        col.markdown(f"""
        <div class="metric-box">
            <div class="stat-number">{val}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### System Architecture")

    arch = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    MINI SEARCH ENGINE                       │
    ├─────────────────────────────────────────────────────────────┤
    │  INPUT LAYER                                                │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
    │  │  20 .txt    │   │  User       │   │  Query      │        │
    │  │  Documents  │   │  Interface  │   │  Input      │        │
    │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘        │
    │         │                 │                 │               │ 
    │  PREPROCESSING PIPELINE                     │               │
    │  Case Fold → Tokenize → Remove Stopwords    │               │
    │  → Stem/Lemmatize → Filter Short Terms      │               │
    │                                             │               │
    │  INDEX LAYER                    QUERY LAYER                 │
    │  ┌───────────┐  ┌───────────┐  ┌───────────┐                │
    │  │ Incidence │  │ Inverted  │  │ Positional│                │
    │  │  Matrix   │  │  Index    │  │  Index    │                │
    │  └───────────┘  └───────────┘  └───────────┘                │
    │                                                             │
    │  RETRIEVAL LAYER                                            │
    │  Boolean (AND/OR/NOT) │ Phrase Query │ TF-IDF Ranking       │
    │                                                             │
    │  BONUS FEATURES                                             │
    │  Wildcard │ Spell Correct │ Query Expand │ Snippets         │
    └─────────────────────────────────────────────────────────────┘
    """
    st.code(arch, language=None)

    if st.session_state['docs_loaded']:
        st.markdown("### Document Collection")
        docs = st.session_state['documents']
        rows = []
        for doc_id, text in docs.items():
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            title = lines[0] if lines else doc_id
            rows.append({
                'Doc ID': doc_id,
                'Title': title,
                'Characters': len(text),
                'Words': len(text.split())
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



elif page == "Preprocessing":
    require_indexes()
    st.header("Document Preprocessing")

    docs = st.session_state['documents']
    doc_choice = st.selectbox("Select a document to inspect:", list(docs.keys()))

    if doc_choice:
        raw_text = docs[doc_choice]
        _, steps = preprocess(
            raw_text,
            use_stemming=st.session_state['use_stemming'],
            use_lemmatization=st.session_state['use_lemma'],
            show_steps=True
        )

        st.markdown("### Step-by-Step Preprocessing")
        step_labels = {
            'original': 'Original Text',
            'case_folded': 'After Case Folding',
            'no_punctuation': 'After Punctuation Removal',
            'tokenized': 'After Tokenization',
            'stopwords_removed': 'After Stopword Removal',
            'stemmed': 'After Stemming',
            'lemmatized': 'After Lemmatization',
            'final_tokens': 'Final Tokens'
        }

        for step_key, label in step_labels.items():
            if step_key in steps:
                with st.expander(label, expanded=(step_key in ['original', 'final_tokens'])):
                    val = steps[step_key]
                    if isinstance(val, list):
                        st.write(' | '.join(val[:30]) + ('...' if len(val) > 30 else ''))
                    else:
                        st.write(str(val)[:500])

        st.markdown("---")
        st.markdown("### Stemming vs Lemmatization Comparison")
        comparison = compare_stemming_lemmatization(raw_text)
        rows = []
        for entry in comparison['comparison_table'][:30]:
            rows.append({
                'Original': entry['original'],
                'Stemmed (Porter)': entry['stemmed'],
                'Lemmatized (WordNet)': entry['lemmatized'],
                'Different?': 'Yes' if entry['different'] else 'Same'
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



elif page == "Incidence Matrix":
    require_indexes()
    st.header("Incidence Matrix")

    im: IncidenceMatrix = st.session_state['inc_matrix']
    stats = im.get_stats()

    col1, col2, col3, col4 = st.columns(4)
    for col, k, lbl in zip(
        [col1, col2, col3, col4],
        ['vocabulary_size', 'num_documents', 'matrix_shape', 'sparsity'],
        ['Vocabulary', 'Documents', 'Matrix Shape', 'Sparsity']
    ):
        col.metric(lbl, str(stats[k]))

    st.markdown("### Full Incidence Matrix (All Terms)")
    sample = im.get_sample(n_terms=stats['vocabulary_size'], n_docs=stats['num_documents'])
    st.dataframe(sample, use_container_width=True)

    st.markdown("---")
    st.markdown("### Boolean Query on Incidence Matrix")
    query_im = st.text_input("Enter Boolean query (e.g. 'machine AND learning', 'data OR mining', 'NOT python')", key="im_query")
    if st.button("Search with Incidence Matrix") and query_im:
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
        results = im.boolean_query(query_im, preprocessor)
        if results:
            st.success(f"Found {len(results)} document(s): {', '.join(results)}")
            for doc_id in results:
                title = doc_title(doc_id, st.session_state['documents'])
                st.markdown(f"- **{doc_id}**: {title}")
        else:
            st.warning("No documents found.")



elif page == "Inverted Index":
    require_indexes()
    st.header("Inverted Index")

    ii: InvertedIndex = st.session_state['inv_index']
    stats = ii.get_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Terms", stats['total_terms'])
    col2.metric("Documents", stats['total_documents'])
    col3.metric("Avg Postings/Term", stats['avg_postings_per_term'])

    st.markdown("### Full Inverted Index (All Terms)")
    entries = ii.get_sample_entries(stats['total_terms'])
    st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Term Lookup")
    look_term = st.text_input("Look up a term:", key="ii_lookup")
    if look_term:
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
        processed = preprocessor(look_term)
        if processed:
            result = ii.search(processed[0])
            if result['df'] > 0:
                st.success(f"**'{processed[0]}'** → df={result['df']} | Postings: {', '.join(result['postings'])}")
            else:
                st.warning(f"Term '{processed[0]}' not found in index.")
                suggestions = suggest_corrections(processed[0], list(ii.index.keys()))
                if suggestions:
                    st.info("Did you mean: " + ", ".join(s['suggestion'] for s in suggestions))

    st.markdown("---")
    st.markdown("### Boolean Query on Inverted Index")
    query_ii = st.text_input("Boolean query (AND/OR/NOT):", key="ii_bool")
    if st.button("Search with Inverted Index") and query_ii:
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
        results = ii.boolean_query(query_ii, preprocessor)
        if results:
            st.success(f"Found {len(results)} document(s):")
            for doc_id in results:
                st.markdown(f"- **{doc_id}**: {doc_title(doc_id, st.session_state['documents'])}")
        else:
            st.warning("No documents matched the query.")



elif page == "Positional Index":
    require_indexes()
    st.header("Positional Index")

    pi: PositionalIndex = st.session_state['pos_index']
    stats = pi.get_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Terms", stats['total_terms'])
    col2.metric("Documents", stats['total_documents'])
    col3.metric("Total Positions Stored", stats['total_positions_stored'])

    st.markdown("### Full Positional Index (All Terms)")
    entries = pi.get_sample_entries(stats['total_terms'])
    st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Phrase Query")
    phrase = st.text_input('Enter a phrase (e.g. "machine learning", "climate change"):', key="pos_phrase")
    if st.button("Search Phrase") and phrase:
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
        phrase_tokens = preprocessor(phrase.replace('"', ''))

        if phrase_tokens:
            st.write(f"Searching for phrase: `{' '.join(phrase_tokens)}`")
            results = pi.phrase_query(phrase_tokens)
            if results:
                st.success(f"Found in {len(results)} document(s):")
                for doc_id in results:
                    title = doc_title(doc_id, st.session_state['documents'])
                    snippet = generate_snippet(st.session_state['documents'][doc_id], phrase_tokens)
                    st.markdown(f"""
                    <div class="result-card">
                        <strong>{doc_id}</strong>: {title}<br>
                        <span class="snippet-text">{snippet}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No documents found containing this exact phrase.")
        else:
            st.warning("Could not process phrase after preprocessing.")

    st.markdown("---")
    st.markdown("### Single Term Position Lookup")
    pos_term = st.text_input("Look up positions of a term:", key="pos_lookup")
    if pos_term:
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
        processed = preprocessor(pos_term)
        if processed:
            result = pi.lookup(processed[0])
            if result['df'] > 0:
                st.success(f"**'{processed[0]}'** appears in {result['df']} document(s):")
                for doc_id, positions in result['postings'].items():
                    st.write(f"  - **{doc_id}**: positions {positions[:10]}{'...' if len(positions) > 10 else ''}")
            else:
                st.warning(f"Term '{processed[0]}' not found.")


elif page == "Ranked Retrieval":
    require_indexes()
    st.header("Ranked Retrieval — TF-IDF & Cosine Similarity")

    ranker: TFIDFRanker = st.session_state['ranker']

    st.markdown("""
    <div class="info-box">
    <strong>Formulas used:</strong><br>
    • TF(t,d) = count(t,d) / total_tokens(d)<br>
    • IDF(t) = log₁₀(N / df(t))<br>
    • TF-IDF(t,d) = TF(t,d) × IDF(t)<br>
    • Cosine(q,d) = (q·d) / (|q|×|d|)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        query_rank = st.text_input("Enter your search query:", placeholder="e.g. machine learning algorithm", key="rank_query")
    with col2:
        top_k = st.number_input("Top K results", min_value=1, max_value=20, value=5)

    use_expansion = st.checkbox("Enable Query Expansion (Bonus)", value=False)

    if st.button("Search & Rank", use_container_width=True) and query_rank:
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
        query_tokens = preprocessor(query_rank)

        if use_expansion:
            expanded = expand_query(query_tokens, preprocessor)
            if set(expanded) != set(query_tokens):
                st.info(f"Expanded query: `{' | '.join(expanded)}`")
            query_tokens = expanded

        results = ranker.rank(query_tokens, top_k=top_k)

        if results:
            st.markdown(f"### Results for: *{query_rank}*")
            st.markdown(f"Found **{len(results)}** relevant documents")

            max_score = results[0]['score'] if results else 1

            for r in results:
                doc_id = r['doc_id']
                score = r['score']
                rank = r['rank']
                title = doc_title(doc_id, st.session_state['documents'])
                snippet = generate_snippet(st.session_state['documents'][doc_id], query_tokens)
                bar_width = int((score / max_score) * 100)

                st.markdown(f"""
                <div class="result-card">
                    <span class="rank-badge">{rank}</span>
                    <strong>{doc_id}</strong> — {title}<br>
                    <small>Cosine Score: <strong>{score:.6f}</strong></small>
                    <div class="score-bar-container">
                        <div style="background: #3b82f6; height: 4px; width: {bar_width}%; border-radius: 2px;"></div>
                    </div>
                    <div class="snippet-text">{snippet}</div>
                </div>
                """, unsafe_allow_html=True)

            # Show TF-IDF breakdown for top result
            st.markdown("---")
            st.markdown(f"### Top TF-IDF Terms in Most Relevant Document ({results[0]['doc_id']})")
            top_terms = ranker.get_top_terms(results[0]['doc_id'], n=15)
            df_terms = pd.DataFrame(top_terms)
            st.bar_chart(df_terms.set_index('term')['tfidf'])

        else:
            st.warning("No relevant documents found.")
            preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
            ii: InvertedIndex = st.session_state['inv_index']
            suggestions = []
            for t in query_tokens:
                suggestions.extend(suggest_corrections(t, list(ii.index.keys())))
            if suggestions:
                st.info("Did you mean: " + ", ".join(set(s['suggestion'] for s in suggestions[:5])))

    st.markdown("---")
    st.markdown("### Inspect TF-IDF for a Specific Document")
    doc_inspect = st.selectbox("Select document:", list(st.session_state['documents'].keys()), key="doc_tfidf")
    if doc_inspect:
        top_terms = ranker.get_top_terms(doc_inspect, n=20)
        if top_terms:
            st.dataframe(pd.DataFrame(top_terms), use_container_width=True, hide_index=True)


elif page == "Boolean Search":
    require_indexes()
    st.header("Boolean Query Search")

    st.markdown("""
    <div class="info-box">
    <strong>Supported operators:</strong><br>
    • <code>term AND term</code> — documents containing both terms<br>
    • <code>term OR term</code> — documents containing either term<br>
    • <code>NOT term</code> — documents not containing the term<br>
    • Single term — documents containing the term
    </div>
    """, unsafe_allow_html=True)

    bool_method = st.radio("Method:", ["Inverted Index", "Incidence Matrix"], horizontal=True)

    st.markdown("**Try an example:**")
    col1, col2, col3 = st.columns(3)
    if col1.button("machine AND learning"):
        st.session_state['bool_main'] = "machine AND learning"
    if col2.button("data OR mining"):
        st.session_state['bool_main'] = "data OR mining"
    if col3.button("NOT python"):
        st.session_state['bool_main'] = "NOT python"

    # 2. مربع النص الآن سيأخذ قيمته من st.session_state['bool_main'] تلقائياً عند ضغط الأزرار
    bool_query = st.text_input("Boolean Query:", placeholder="e.g. machine AND learning", key="bool_main")

    if st.button("Execute Boolean Query") and bool_query:
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])

        start = time.time()
  
        try:
            if bool_method == "Inverted Index":
                results = st.session_state['inv_index'].boolean_query(bool_query, preprocessor)
            else:
                results = st.session_state['inc_matrix'].boolean_query(bool_query, preprocessor)
            
            elapsed = round((time.time() - start) * 1000, 2)

            st.markdown(f"*Query processed in {elapsed} ms using {bool_method}*")

            if results:
                st.success(f"Found **{len(results)}** document(s) for: `{bool_query}`")
                for doc_id in results:
                    title = doc_title(doc_id, st.session_state['documents'])
                    st.markdown(f"""
                    <div class="result-card">
                        <strong>{doc_id}</strong>: {title}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No documents matched the query.")
                
        except Exception as e:
            
            st.error(f"An error occurred while executing the query: {str(e)}")
            st.info("Check your 'boolean_query' function logic in the backend classes.")


elif page == "Bonus Features":
    require_indexes()
    st.header("Bonus Features")

    tab1, tab2, tab3, tab4 = st.tabs(["Wildcard", "Spell Check", "Query Expansion", "Snippets"])

    ii: InvertedIndex = st.session_state['inv_index']
    vocabulary = list(ii.index.keys())

    with tab1:
        st.markdown("### Wildcard Query Processing")
        st.markdown("Use `*` as a wildcard. Examples: `comput*`, `*tion`, `mac*ine`")
        wildcard_input = st.text_input("Enter wildcard pattern:", placeholder="e.g. learn*")
        if st.button("Search Wildcard") and wildcard_input:
            matches = wildcard_search(wildcard_input, vocabulary)
            if matches:
                st.success(f"Found {len(matches)} matching terms:")
                cols = st.columns(4)
                for i, term in enumerate(matches[:40]):
                    cols[i % 4].markdown(f"<span class='tag'>{term}</span>", unsafe_allow_html=True)
                if len(matches) > 40:
                    st.write(f"...and {len(matches)-40} more")

                all_docs = set()
                for term in matches:
                    all_docs.update(ii.get_postings(term))
                if all_docs:
                    st.markdown(f"**Documents containing wildcard matches:** {', '.join(sorted(all_docs))}")
            else:
                st.warning("No matching terms found.")

    with tab2:
        st.markdown("### Spelling Correction (Edit Distance)")
        spell_input = st.text_input("Enter a possibly misspelled term:", placeholder="e.g. lerning, machin")
        max_dist = st.slider("Max Edit Distance", 1, 4, 2)
        if st.button("Get Suggestions") and spell_input:
            suggestions = suggest_corrections(spell_input.lower(), vocabulary, max_distance=max_dist)
            if suggestions:
                st.success(f"Suggestions for '{spell_input}':")
                df_sug = pd.DataFrame(suggestions)
                st.dataframe(df_sug, use_container_width=True, hide_index=True)
            else:
                st.warning("No suggestions found within the edit distance limit.")

    with tab3:
        st.markdown("### Query Expansion (Synonym-Based)")
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
        expand_input = st.text_input("Enter query to expand:", placeholder="e.g. ai data")
        if st.button("Expand Query") and expand_input:
            original_tokens = preprocessor(expand_input)
            expanded_tokens = expand_query(original_tokens, preprocessor)
            new_tokens = [t for t in expanded_tokens if t not in original_tokens]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original tokens:**")
                for t in original_tokens:
                    st.markdown(f"<span class='tag'>{t}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown("**Expansion added:**")
                for t in new_tokens:
                    st.markdown(f"<span class='tag' style='background:#f0fdf4;color:#166534;border-color:#bbf7d0'>{t}</span>", unsafe_allow_html=True)

            results = st.session_state['ranker'].rank(expanded_tokens, top_k=5)
            if results:
                st.markdown("**Results with expanded query:**")
                for r in results:
                    st.write(f"**#{r['rank']}** {r['doc_id']}: {doc_title(r['doc_id'], st.session_state['documents'])} (score: {r['score']:.4f})")


elif page == "Evaluation":
    require_indexes()
    st.header("Retrieval Evaluation — Precision, Recall, F1")

    st.markdown("""
    <div class="info-box">
    Define which documents you consider <strong>relevant</strong> for a query,
    then compare against what the system retrieved.
    Precision = TP / Retrieved | Recall = TP / Relevant | F1 = 2·P·R / (P+R)
    </div>
    """, unsafe_allow_html=True)

    eval_query = st.text_input("Query:", placeholder="e.g. artificial intelligence machine learning")
    all_doc_ids = sorted(st.session_state['documents'].keys())
    relevant_docs = st.multiselect("Mark which documents are TRULY RELEVANT:", all_doc_ids)

    eval_method = st.radio("Retrieval Method:", ["TF-IDF Ranked", "Boolean (AND)", "Boolean (OR)"], horizontal=True)
    top_k_eval = st.slider("Retrieve top K documents:", 1, 20, 5)

    if st.button("Run Evaluation") and eval_query and relevant_docs:
        preprocessor = get_preprocessor(st.session_state['use_stemming'], st.session_state['use_lemma'])
        query_tokens = preprocessor(eval_query)

        if eval_method == "TF-IDF Ranked":
            results = st.session_state['ranker'].rank(query_tokens, top_k=top_k_eval)
            retrieved = [r['doc_id'] for r in results]
        elif eval_method == "Boolean (AND)":
            retrieved = st.session_state['inv_index'].boolean_query(eval_query + " AND ".join([""] * 0), preprocessor)
        else:
            retrieved = st.session_state['inv_index'].boolean_query(" OR ".join(eval_query.split()), preprocessor)

        metrics = compute_metrics(retrieved, relevant_docs)

        col1, col2, col3, col4 = st.columns(4)
        for col, metric, val in zip(
            [col1, col2, col3, col4],
            ['Precision', 'Recall', 'F1-Score', 'True Positives'],
            [metrics['precision'], metrics['recall'], metrics['f1'], metrics['true_positives']]
        ):
            col.metric(metric, f"{val:.4f}" if isinstance(val, float) else val)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Retrieved Documents:**")
            for d in retrieved:
                match = "[Match]" if d in relevant_docs else "[Miss]"
                st.write(f"{match} {d}: {doc_title(d, st.session_state['documents'])}")
        with col2:
            st.markdown("**Relevant Documents (Ground Truth):**")
            for d in relevant_docs:
                found = "[Found]" if d in retrieved else "[Missed]"
                st.write(f"{found} {d}: {doc_title(d, st.session_state['documents'])}")

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center><small>Mini Search Engine — Information Management Systems</small></center>",
    unsafe_allow_html=True
)