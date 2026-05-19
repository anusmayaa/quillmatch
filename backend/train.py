import json
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix

from gutenberg import fetch_all_authors
from features import extract_features

MODEL_PATH      = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
SCALER_PATH     = 'scaler.pkl'
PROFILES_PATH   = 'author_profiles.json'

TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE  = (1, 3)
TFIDF_MIN_DF       = 2


def build_feature_matrix(chunks):
    feature_dicts = []
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"    Extracting features: {i}/{len(chunks)} chunks")
        feature_dicts.append(extract_features(chunk))
    df = pd.DataFrame(feature_dicts)
    return df.values, list(df.columns)


def compute_author_profiles(author_labels, feature_matrix, feature_names):
    profiles = {}
    for author in set(author_labels):
        indices     = [i for i, l in enumerate(author_labels) if l == author]
        author_mean = feature_matrix[indices].mean(axis=0)
        profiles[author] = {
            feature_names[j]: float(author_mean[j])
            for j in range(len(feature_names))
        }
    return profiles


def normalise_profiles(profiles):
    feature_names = list(list(profiles.values())[0].keys())
    normalised    = {author: {} for author in profiles}
    for feature in feature_names:
        values      = [profiles[a][feature] for a in profiles]
        min_val     = min(values)
        max_val     = max(values)
        value_range = max_val - min_val
        for author in profiles:
            if value_range == 0:
                normalised[author][feature] = 0.5
            else:
                scaled = (profiles[author][feature] - min_val) / value_range
                normalised[author][feature] = round(0.15 + scaled * 0.70, 4)
    return normalised


def train():
    print("=" * 60)
    print("STEP 1: Fetching chunks from Gutenberg")
    print("=" * 60)
    author_data = fetch_all_authors()

    if not author_data:
        print("[!] No data fetched.")
        return

    all_chunks, all_labels, all_book_ids = [], [], []
    for author, chunk_tuples in author_data.items():
        for chunk_text, book_id in chunk_tuples:
            all_chunks.append(chunk_text)
            all_labels.append(author)
            all_book_ids.append(book_id)

    all_book_ids = np.array(all_book_ids)

    print(f"\nTotal samples: {len(all_chunks)}")
    for author, chunk_tuples in author_data.items():
        print(f"  {author}: {len(chunk_tuples)} chunks")

    # ── STYLE FEATURES (display only — not used in classifier) ─────────
    print("\n" + "=" * 60)
    print("STEP 2: Extracting style features for radar chart")
    print("=" * 60)
    feature_matrix, feature_names = build_feature_matrix(all_chunks)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # ── TF-IDF (this is what the classifier uses) ──────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Building TF-IDF vectors")
    print("=" * 60)
    vectorizer = TfidfVectorizer(
        max_features = TFIDF_MAX_FEATURES,
        ngram_range  = TFIDF_NGRAM_RANGE,
        min_df       = TFIDF_MIN_DF,
        sublinear_tf = True,
        analyzer     = 'word',
    )
    # Classifier trains on TF-IDF only
    # Style features power the radar chart and explanation — not this
    tfidf_matrix = vectorizer.fit_transform(all_chunks)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # ── BOOK LEVEL HOLDOUT ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Book-level holdout evaluation")
    print("=" * 60)

    train_idx, test_idx = [], []
    for author, chunk_tuples in author_data.items():
        author_indices = [i for i, l in enumerate(all_labels) if l == author]
        author_books   = list(dict.fromkeys(
            all_book_ids[i] for i in author_indices
        ))
        # Use middle book as test set
        test_book = author_books[len(author_books) // 2]
        for i in author_indices:
            if all_book_ids[i] == test_book:
                test_idx.append(i)
            else:
                train_idx.append(i)

    X_train = tfidf_matrix[train_idx]
    X_test  = tfidf_matrix[test_idx]
    y_train = [all_labels[i] for i in train_idx]
    y_test  = [all_labels[i] for i in test_idx]

    print(f"Train: {len(train_idx)} chunks")
    print(f"Test:  {len(test_idx)} chunks")

    # Show test set balance
    from collections import Counter
    test_counts = Counter(y_test)
    for author, count in sorted(test_counts.items()):
        print(f"  {author:<25} {count} test chunks")

    base_eval = LinearSVC(
        C            = 1.0,
        class_weight = 'balanced',
        random_state = 42,
        max_iter     = 5000,
    )
    svm_eval = CalibratedClassifierCV(base_eval, cv=2, method='isotonic')
    print("\nTraining evaluation model...")
    svm_eval.fit(X_train, y_train)

    y_pred   = svm_eval.predict(X_test)
    test_acc = np.mean(np.array(y_pred) == np.array(y_test))

    print(f"\nReal holdout accuracy: {test_acc:.1%}")
    print("\nPer-author breakdown:")
    print(classification_report(y_test, y_pred))

    # ── FULL RETRAIN FOR DEPLOYMENT ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Retraining on full data for deployment")
    print("=" * 60)
    base_full = LinearSVC(
        C            = 1.0,
        class_weight = 'balanced',
        random_state = 42,
        max_iter     = 5000,
    )
    svm = CalibratedClassifierCV(base_full, cv=2, method='isotonic')
    svm.fit(tfidf_matrix, all_labels)
    print("Full model trained.")

    # ── AUTHOR PROFILES (style features for display) ───────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Computing author profiles")
    print("=" * 60)
    raw_profiles  = compute_author_profiles(all_labels, feature_matrix, feature_names)
    norm_profiles = normalise_profiles(raw_profiles)

    with open(PROFILES_PATH, 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'raw':           raw_profiles,
            'normalised':    norm_profiles,
        }, f, indent=2)
    print(f"Profiles saved → {PROFILES_PATH}")

    # Scaler saved but not used in prediction
    # Kept for compatibility with main.py
    scaler = StandardScaler()
    scaler.fit(feature_matrix)

    print("\n" + "=" * 60)
    print("STEP 7: Saving artifacts")
    print("=" * 60)
    with open(MODEL_PATH, 'w+b')      as f: pickle.dump(svm,        f)
    with open(VECTORIZER_PATH, 'w+b') as f: pickle.dump(vectorizer, f)
    with open(SCALER_PATH, 'w+b')     as f: pickle.dump(scaler,     f)

    print(f"Saved: {MODEL_PATH}, {VECTORIZER_PATH}, {SCALER_PATH}")
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Real holdout accuracy (book-level): {test_acc:.1%}")
    print("=" * 60)


if __name__ == '__main__':
    train()