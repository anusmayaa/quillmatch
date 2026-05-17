import json
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack, csr_matrix

from gutenberg import fetch_all_authors
from features import extract_features


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MODEL_PATH    = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
SCALER_PATH   = 'scaler.pkl'
PROFILES_PATH = 'author_profiles.json'

# TF-IDF settings
# max_features: only keep the 3000 most useful words
# ngram_range: look at single words AND two-word pairs ("baker street")
# min_df: ignore words that appear in fewer than 3 chunks (probably noise)
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 3


def build_feature_matrix(chunks: list[str]) -> np.ndarray:
    """
    Runs extract_features() on every chunk and stacks
    the results into a 2D numpy array.

    Shape: (num_chunks, 11)

    Each row = one chunk's style feature vector.
    Each column = one of our 11 features.

    Common mistake: extract_features returns a dict —
    we must extract values in a CONSISTENT ORDER every time.
    Using sorted(keys) guarantees this.
    """
    feature_dicts = []
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"    Extracting features: {i}/{len(chunks)} chunks")
        features = extract_features(chunk)
        feature_dicts.append(features)

    # Convert list of dicts to DataFrame first —
    # this guarantees column order stays consistent
    df = pd.DataFrame(feature_dicts)

    # Store the column order so we can use it consistently
    # This becomes important in main.py during prediction
    return df.values, list(df.columns)


def compute_author_profiles(
    author_labels: list[str],
    feature_matrix: np.ndarray,
    feature_names: list[str]
) -> dict:
    """
    For each author, compute the mean of all their chunk feature vectors.
    This gives us one representative vector per author.

    This is what gets stored in author_profiles.json and used
    by the radar chart and explanation generator.

    Note: we use the RAW (unscaled) features here so the values
    are human-readable in the frontend (0.18 not -1.23).
    """
    profiles = {}
    unique_authors = list(set(author_labels))

    for author in unique_authors:
        # Find all row indices belonging to this author
        indices = [i for i, label in enumerate(author_labels) if label == author]

        # Slice those rows and compute column-wise mean
        author_rows = feature_matrix[indices]
        author_mean = author_rows.mean(axis=0)

        # Store as dict with feature names as keys
        profiles[author] = {
            feature_names[j]: float(author_mean[j])
            for j in range(len(feature_names))
        }

    return profiles


def normalise_profiles(profiles: dict) -> dict:
    """
    Normalises author profile values to 0-1 range for radar chart display.

    We compute min/max across ALL authors for each feature,
    then normalise each author's value within that range.

    This is separate from StandardScaler (which is for the SVM).
    This normalisation is purely for visualisation.
    """
    # Collect all feature names from first author
    first_author   = list(profiles.keys())[0]
    feature_names  = list(profiles[first_author].keys())

    normalised = {author: {} for author in profiles}

    for feature in feature_names:
        # Get this feature's value across all authors
        values = [profiles[author][feature] for author in profiles]
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val

        for author in profiles:
            raw_val = profiles[author][feature]
            if value_range == 0:
                # All authors identical on this feature — set to 0.5
                normalised[author][feature] = 0.5
            else:
                normalised[author][feature] = (raw_val - min_val) / value_range

    return normalised


def train():
    """
    Main training function. Steps:
    1. Fetch all author text chunks from Gutenberg
    2. Extract style features from every chunk
    3. Build TF-IDF vectors from every chunk
    4. Combine style features + TF-IDF into one matrix
    5. Scale the style features
    6. Train SVM classifier
    7. Evaluate with cross-validation
    8. Save model, vectorizer, scaler, and author profiles
    """

    # ── STEP 1: FETCH DATA ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Fetching author text chunks from Gutenberg")
    print("=" * 60)
    # Returns dict: { "Jane Austen": [chunk1, chunk2, ...], ... }
    author_data = fetch_all_authors()

    if not author_data:
        print("[!] No data fetched. Check your internet connection.")
        return

    # ── STEP 2: BUILD LABELS AND CHUNK LISTS ─────────────────────────
    # Flatten everything into parallel lists:
    # all_chunks[i] and all_labels[i] always correspond to each other
    all_chunks = []
    all_labels = []

    for author, chunks in author_data.items():
        all_chunks.extend(chunks)
        all_labels.extend([author] * len(chunks))

    print(f"\nTotal training samples: {len(all_chunks)}")
    for author, chunks in author_data.items():
        print(f"  {author}: {len(chunks)} chunks")

    # ── STEP 3: EXTRACT STYLE FEATURES ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Extracting style features from all chunks")
    print("=" * 60)
    feature_matrix, feature_names = build_feature_matrix(all_chunks)
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    # Should be roughly (total_chunks, 11)

    # ── STEP 4: BUILD TF-IDF MATRIX ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Building TF-IDF vectors")
    print("=" * 60)

    # fit_transform does two things in one call:
    # 1. fit   — learn the vocabulary from all chunks
    # 2. transform — convert every chunk into a TF-IDF vector
    vectorizer = TfidfVectorizer(
        max_features = TFIDF_MAX_FEATURES,
        ngram_range  = TFIDF_NGRAM_RANGE,
        min_df       = TFIDF_MIN_DF,
        sublinear_tf = True,  # apply log(tf) — reduces impact of very frequent words
    )
    tfidf_matrix = vectorizer.fit_transform(all_chunks)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    # Should be roughly (total_chunks, 3000)

    # ── STEP 5: SCALE STYLE FEATURES ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Scaling style features")
    print("=" * 60)

    # StandardScaler: transforms each feature to mean=0, std=1
    # fit_transform on training data — we save this scaler
    # so we apply the EXACT SAME transformation at prediction time
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    print("Style features scaled (mean=0, std=1)")

    # ── STEP 6: COMBINE STYLE FEATURES + TF-IDF ──────────────────────
    # hstack = horizontal stack (add columns side by side)
    # style features: shape (n, 11)
    # tfidf matrix:   shape (n, 3000)
    # combined:       shape (n, 3011)
    #
    # We convert feature_matrix_scaled to sparse format first
    # because hstack requires both matrices to be sparse
    feature_sparse = csr_matrix(feature_matrix_scaled)
    combined_matrix = hstack([feature_sparse, tfidf_matrix])
    print(f"Combined matrix shape: {combined_matrix.shape}")

    # ── STEP 7: TRAIN SVM ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Training SVM classifier")
    print("=" * 60)

    # kernel='rbf'  — Radial Basis Function
    #   This is the most powerful kernel for this type of problem.
    #   It can draw curved boundaries between author clusters,
    #   unlike 'linear' which can only draw straight lines.
    #
    # C=5           — regularisation strength
    #   Higher C = model tries harder to correctly classify training data
    #   but risks overfitting. C=5 is a good middle ground for text tasks.
    #
    # probability=True — enables confidence scores via Platt Scaling
    #   This makes training ~5x slower but we need it for confidence %
    #
    # class_weight='balanced' — compensates if some authors have more chunks
    #   Automatically weights each author inversely to their chunk count

    svm = SVC(
        kernel       = 'rbf',
        C            = 5,
        probability  = True,
        class_weight = 'balanced',
        random_state = 42,
    )

    print("Training SVM... (this takes 2-5 minutes with probability=True)")
    svm.fit(combined_matrix, all_labels)
    print("Training complete.")

    # ── STEP 8: CROSS-VALIDATION ──────────────────────────────────────
    # Cross-validation = the honest way to measure accuracy
    #
    # We split data into 5 folds. For each fold:
    # - Train on 4/5 of the data
    # - Test on the remaining 1/5
    # - Record accuracy
    # Average the 5 scores = our real accuracy estimate
    #
    # This is more trustworthy than just checking training accuracy
    # because it tests on data the model hasn't seen
    print("\n" + "=" * 60)
    print("STEP 7: Cross-validation (5-fold)")
    print("=" * 60)
    print("Running... (another 2-3 minutes)")

    cv_scores = cross_val_score(svm, combined_matrix, all_labels, cv=5)
    print(f"CV Accuracy scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"Mean accuracy:      {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # ── STEP 9: COMPUTE AND SAVE AUTHOR PROFILES ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Computing author profiles")
    print("=" * 60)

    # Raw profiles (for explanation generator)
    raw_profiles = compute_author_profiles(
        all_labels, feature_matrix, feature_names
    )

    # Normalised profiles (for radar chart — values 0 to 1)
    normalised_profiles = normalise_profiles(raw_profiles)

    # Store both in one JSON file
    profiles_to_save = {
        'feature_names': feature_names,
        'raw':           raw_profiles,
        'normalised':    normalised_profiles,
    }

    with open(PROFILES_PATH, 'w') as f:
        json.dump(profiles_to_save, f, indent=2)
    print(f"Author profiles saved to {PROFILES_PATH}")

    # ── STEP 10: SAVE MODEL, VECTORIZER, SCALER ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Saving model, vectorizer and scaler")
    print("=" * 60)

    with open(MODEL_PATH, 'w+b') as f:
        pickle.dump(svm, f)
    print(f"Model saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, 'w+b') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    with open(SCALER_PATH, 'w+b') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Mean CV accuracy: {cv_scores.mean():.1%}")
    print("=" * 60)


if __name__ == '__main__':
    train()