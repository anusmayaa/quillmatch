# QuillMatch

A literary authorship classifier that analyses your writing style and matches it to one of 7 classic authors. Paste any prose sample and get a confidence-scored match with a per-feature style breakdown.

![QuillMatch UI](https://img.shields.io/badge/stack-React%20%2B%20FastAPI%20%2B%20scikit--learn-C8A050?style=flat&labelColor=1C150D)

---

## How it works

1. **Training data** — Text chunks (~500 words each) are fetched from Project Gutenberg for 7 authors
2. **Feature extraction** — 11 stylometric features are extracted from every chunk (sentence length, vocab richness, punctuation density, etc.)
3. **TF-IDF vectors** — 3000-feature TF-IDF matrix captures vocabulary patterns
4. **SVM classifier** — A Support Vector Machine with RBF kernel is trained on the combined feature matrix
5. **Prediction** — User text is processed through the same pipeline and matched to the closest author

## Authors

| Author | Era |
|---|---|
| Jane Austen | 1775 – 1817 |
| Charles Dickens | 1812 – 1870 |
| Mark Twain | 1835 – 1910 |
| Oscar Wilde | 1854 – 1900 |
| Virginia Woolf | 1882 – 1941 |
| Edgar Allan Poe | 1809 – 1849 |
| Arthur Conan Doyle | 1859 – 1930 |

## Style features

| # | Feature | Description |
|---|---|---|
| 1 | avg_sentence_length | Mean words per sentence |
| 2 | vocab_richness | Type-token ratio (unique / total words) |
| 3 | punctuation_density | Punctuation characters per total characters |
| 4 | avg_word_length | Mean characters per word |
| 5 | function_word_ratio | Proportion of function words |
| 6 | sentence_length_variance | Standard deviation of sentence lengths |
| 7 | comma_frequency | Commas per sentence |
| 8 | question_mark_frequency | Question marks per 100 sentences |
| 9 | exclamation_frequency | Exclamations per 100 sentences |
| 10 | paragraph_length | Mean sentences per paragraph |
| 11 | lexical_density | Content words / total words |

## Tech stack

**Backend**
- Python 3.11+
- FastAPI — REST API
- scikit-learn — SVM classifier, TF-IDF vectoriser, StandardScaler
- NLTK — sentence tokenisation, POS tagging
- scipy — sparse matrix operations

**Frontend**
- React 18
- Vanilla CSS — no UI library
- Google Fonts — Playfair Display + EB Garamond

---

## Project structure

```
quillmatch/
├── backend/
│   ├── main.py           # FastAPI app + prediction endpoint
│   ├── train.py          # Model training pipeline
│   ├── features.py       # 11 stylometric feature extractors
│   ├── gutenberg.py      # Project Gutenberg data fetcher
│   ├── model.pkl         # Trained SVM model
│   ├── vectorizer.pkl    # Fitted TF-IDF vectoriser
│   ├── scaler.pkl        # Fitted StandardScaler
│   └── author_profiles.json  # Per-author feature profiles
└── frontend/
    └── src/
        ├── App.js
        ├── api/
        │   └── api.js        # Backend API calls
        └── components/
            ├── TextInput.js  # Input screen
            ├── ResultCard.js # Results screen
            ├── RadarChart.js # SVG radar chart
            └── FeatureBar.js # Feature comparison bars
```

---

## Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install fastapi uvicorn scikit-learn nltk scipy numpy pandas requests
python train.py              # fetches data + trains model (~15 min)
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000)

> Make sure the backend is running on port 8000 before using the frontend.

---

## API

### `POST /predict`

**Request**
```json
{ "text": "Your writing sample here..." }
```

**Response**
```json
{
  "author": "Jane Austen",
  "confidence": 78.4,
  "explanation": "Your writing closely matches Jane Austen in...",
  "user_features": { "avg_sentence_length": 0.85, ... },
  "author_features": { "avg_sentence_length": 0.85, ... },
  "all_scores": { "Jane Austen": 78.4, "Charles Dickens": 9.1, ... }
}
```

### `GET /health`
Returns `{ "status": "ok" }` — use to check the server is running.
