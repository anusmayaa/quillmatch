import json
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix

from features import extract_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('model.pkl', 'rb')      as f: model      = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)
with open('scaler.pkl', 'rb')     as f: scaler     = pickle.load(f)
with open('author_profiles.json') as f: profiles   = json.load(f)

FEATURE_NAMES    = profiles['feature_names']
RAW_PROFILES     = profiles['raw']
NORM_PROFILES    = profiles['normalised']

MIN_WORD_COUNT = 100

FEATURE_DESCRIPTIONS = {
    'avg_sentence_length': {
        'high': 'long, flowing sentences',
        'low':  'short, punchy sentences',
    },
    'vocab_richness': {
        'high': 'a rich and varied vocabulary',
        'low':  'deliberate repetition of simple words',
    },
    'punctuation_density': {
        'high': 'heavy use of punctuation',
        'low':  'sparse punctuation',
    },
    'avg_word_length': {
        'high': 'longer, more complex words',
        'low':  'short, simple words',
    },
    'function_word_ratio': {
        'high': 'a high proportion of function words',
        'low':  'a low proportion of function words',
    },
    'sentence_length_variance': {
        'high': 'varied sentence rhythm — mixing short and long',
        'low':  'very consistent sentence rhythm',
    },
    'comma_frequency': {
        'high': 'frequent comma-chained clauses',
        'low':  'few commas — preferring full stops',
    },
    'question_mark_frequency': {
        'high': 'frequent rhetorical questions',
        'low':  'almost no questions — declarative prose',
    },
    'exclamation_frequency': {
        'high': 'emotionally expressive punctuation',
        'low':  'restrained, unemotional punctuation',
    },
    'paragraph_length': {
        'high': 'long, dense paragraphs',
        'low':  'short, broken paragraphs',
    },
    'lexical_density': {
        'high': 'information-dense prose',
        'low':  'conversational, flowing prose',
    },
}

AUTHOR_STYLE_LABELS = {
    'Jane Austen':        'her sharp social observation',
    'Charles Dickens':    'his sprawling Victorian prose',
    'Mark Twain':         'his conversational American voice',
    'Oscar Wilde':        'his witty, epigrammatic style',
    'Virginia Woolf':     'her stream-of-consciousness flow',
    'Edgar Allan Poe':    'his dark, Gothic intensity',
    'Arthur Conan Doyle': 'his precise, analytical clarity',
}


def normalise_user_features(raw_features: dict) -> dict:
    normalised = {}
    for feature in FEATURE_NAMES:
        author_values = [RAW_PROFILES[a][feature] for a in RAW_PROFILES]
        min_val       = min(author_values) * 0.7
        max_val       = max(author_values) * 1.3
        raw_val       = raw_features.get(feature, 0.0)
        value_range   = max_val - min_val
        if value_range == 0:
            normalised[feature] = 0.5
        else:
            scaled = (raw_val - min_val) / value_range
            normalised[feature] = round(
                max(0.05, min(0.95, 0.15 + scaled * 0.70)), 4
            )
    return normalised


def build_explanation(author, user_features, author_features):
    gaps = []
    for feature in FEATURE_NAMES:
        user_val   = user_features.get(feature, 0.0)
        author_val = author_features.get(feature, 0.0)
        gap        = abs(user_val - author_val)
        direction  = 'high' if user_val > 0.5 else 'low'
        gaps.append((gap, feature, direction))

    # Sort SMALLEST gap first — these are the features that MATCH
    gaps.sort(reverse=False)
    matching = gaps[:3]   # top 3 closest features

    # Also grab the 1 biggest gap — the main difference
    biggest_diff = gaps[-1]

    descriptions = []
    for _, feature, direction in matching:
        desc = FEATURE_DESCRIPTIONS.get(feature, {}).get(direction, feature)
        descriptions.append(desc)

    style_label = AUTHOR_STYLE_LABELS.get(author, f"{author}'s style")

    sentence1 = (
        f"Your writing closely matches {author} in "
        f"{descriptions[0]} and {descriptions[1]}."
    )
    sentence2 = (
        f"This aligns with {style_label}. "
        f"The strongest shared signal is {descriptions[2]}."
    )

    return f"{sentence1} {sentence2}"


class PredictRequest(BaseModel):
    text: str


@app.post('/predict')
def predict(request: PredictRequest):
    text = request.text.strip()

    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Text too short. Minimum {MIN_WORD_COUNT} words, got {word_count}."
        )

    # Style features for display only
    raw_features = extract_features(text)

    # Classifier uses TF-IDF only — same as training
    tfidf_vector   = vectorizer.transform([text])
    probabilities  = model.predict_proba(tfidf_vector)[0]
    classes        = model.classes_
    confidence_map = {
        cls: round(float(prob) * 100, 1)
        for cls, prob in zip(classes, probabilities)
    }

    matched_author = max(confidence_map, key=confidence_map.get)
    confidence     = confidence_map[matched_author]

    user_norm    = normalise_user_features(raw_features)
    author_norm  = NORM_PROFILES.get(matched_author, {})
    explanation  = build_explanation(matched_author, user_norm, author_norm)

    return {
        'author':          matched_author,
        'confidence':      confidence,
        'explanation':     explanation,
        'user_features':   user_norm,
        'author_features': author_norm,
        'all_scores':      confidence_map,
    }

@app.get('/authors')
def get_authors():
    return {'authors': list(RAW_PROFILES.keys())}


@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/debug')
def debug():
    import os
    return {
        'model_type':    str(type(model)),
        'classes':       list(model.classes_),
        'working_dir':   os.getcwd(),
        'profiles_keys': list(RAW_PROFILES.keys()),
    }