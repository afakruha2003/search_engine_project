#  Mini Search Engine

**Course:** Information Management Systems (SENG328)  
**Language:** Python 3.10+  
**Interface:** Streamlit Web App

---

##  Project Structure

```
search_engine_project/
├── documents/          ← 20 plain-text documents
│   ├── doc01.txt       (Artificial Intelligence)
│   ├── doc02.txt       (Climate Change)
│   ├── ...
│   └── doc20.txt       (Social Media)
├── main.py             ← Streamlit web interface (run this)
├── preprocessing.py    ← Case folding, tokenization, stopwords, stemming/lemmatization
├── incidence_matrix.py ← Binary term-document matrix + Boolean queries
├── inverted_index.py   ← Inverted index + Boolean AND/OR/NOT with posting merge
├── positional_index.py ← Positional index + phrase query matching
├── ranking.py          ← TF-IDF weighting + Cosine Similarity ranking
├── utils.py            ← BONUS: Wildcard, Spell Correction, Query Expansion, Snippets, Metrics
├── README.md
└── requirements.txt
```

---

##  Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run main.py
```

Then open your browser at: **http://localhost:8501**

---

##  How to Use

1. **Sidebar → Enter folder path** (default: `documents`)
2. **Choose preprocessing method**: Stemming or Lemmatization
3. **Click "Load & Build Indexes"** — all 4 indexes are built automatically
4. **Navigate using the sidebar** to explore each feature:

| Page | Description |
|------|-------------|
|  Home | System overview, document list, architecture diagram |
|  Preprocessing | Step-by-step view of normalization pipeline |
|  Incidence Matrix | Binary matrix display + Boolean query |
|  Inverted Index | Posting lists + Boolean AND/OR/NOT search |
|  Positional Index | Term positions + exact phrase queries |
|  Ranked Retrieval | TF-IDF scoring + Cosine Similarity ranking |
|  Boolean Search | Combined Boolean query interface |
|  Bonus Features | Wildcard, Spell Check, Query Expansion, Snippets |
|  Evaluation | Precision, Recall, F1-Score computation |

---

##  Example Queries

### Boolean Queries
```
machine AND learning
climate OR energy
NOT python
artificial AND intelligence
```

### Phrase Queries (Positional Index)
```
machine learning
climate change
deep learning
renewable energy
neural network
```

### Ranked Retrieval Queries
```
artificial intelligence neural network
climate change renewable energy
cybersecurity data protection
space exploration mars
```

### Wildcard Queries
```
comput*
*tion
learn*
mac*ine
```

---

##  Bonus Features Implemented 

-  **Wildcard query processing** — regex-based prefix/suffix/infix matching
-  **Spelling correction** — Levenshtein edit distance
-  **Query expansion** — synonym-based term expansion
-  **Document snippets** — context-aware snippet extraction
-  **Web-based GUI** — Streamlit interface
-  **Evaluation metrics** — Precision, Recall, F1-Score
-  **Stemming vs Lemmatization comparison** — side-by-side table

---

##  Dependencies

```
streamlit
nltk
pandas
numpy
```

---


