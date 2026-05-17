import re
import string
import nltk

# Download these once — NLTK needs them for tokenizing and POS tagging
nltk.download('punkt',        quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab',    quiet=True)

# These are the function words we check against for feature 5
# This list covers the most common English function words
FUNCTION_WORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','by','from','is','was','are','were','be','been',
    'being','have','has','had','do','does','did','will','would',
    'could','should','may','might','shall','can','need','dare',
    'ought','used','it','its','this','that','these','those',
    'i','me','my','we','our','you','your','he','him','his',
    'she','her','they','them','their','what','which','who',
    'not','no','nor','so','yet','both','either','neither',
    'as','if','though','although','because','since','while',
    'about','above','after','before','between','into','through',
    'during','without','within','along','following','across'
])


def extract_features(text: str) -> dict:
    """
    Takes a raw string of text and returns a dictionary
    of 11 style features. All values are raw (not yet normalised —
    normalisation happens in train.py before feeding to the model).
    """

    # ── 1. TOKENISE INTO SENTENCES AND WORDS ──────────────────────────
    # sent_tokenize splits on sentence boundaries (handles Mr. Dr. etc.)
    # word_tokenize splits into individual word tokens
    sentences = nltk.sent_tokenize(text)
    words     = nltk.word_tokenize(text)

    # Filter to only alphabetic words (removes punctuation tokens)
    alpha_words = [w for w in words if w.isalpha()]

    # Guard: if the text is too short, return zeros
    # This prevents division-by-zero errors on tiny inputs
    if len(sentences) == 0 or len(alpha_words) == 0:
        return {k: 0.0 for k in [
            'avg_sentence_length','vocab_richness','punctuation_density',
            'avg_word_length','function_word_ratio','sentence_length_variance',
            'comma_frequency','question_mark_frequency','exclamation_frequency',
            'paragraph_length','lexical_density'
        ]}

    # ── 2. SENTENCE-LEVEL WORD COUNTS ─────────────────────────────────
    # For each sentence, count how many alpha words it contains
    # We use this for avg_sentence_length and sentence_length_variance
    sentence_word_counts = []
    for sent in sentences:
        sent_words = [w for w in nltk.word_tokenize(sent) if w.isalpha()]
        sentence_word_counts.append(len(sent_words))

    # ── FEATURE 1: avg_sentence_length ────────────────────────────────
    avg_sentence_length = sum(sentence_word_counts) / len(sentence_word_counts)

    # ── FEATURE 2: vocab_richness ─────────────────────────────────────
    # type-token ratio: unique words / total words
    # lowercased so "The" and "the" count as the same word
    unique_words  = set(w.lower() for w in alpha_words)
    vocab_richness = len(unique_words) / len(alpha_words)

    # ── FEATURE 3: punctuation_density ───────────────────────────────
    # Count every punctuation character in the raw text
    punct_count       = sum(1 for ch in text if ch in string.punctuation)
    punctuation_density = punct_count / len(text) if len(text) > 0 else 0.0

    # ── FEATURE 4: avg_word_length ────────────────────────────────────
    avg_word_length = sum(len(w) for w in alpha_words) / len(alpha_words)

    # ── FEATURE 5: function_word_ratio ───────────────────────────────
    # Check each word (lowercased) against our function word set
    func_count          = sum(1 for w in alpha_words if w.lower() in FUNCTION_WORDS)
    function_word_ratio = func_count / len(alpha_words)

    # ── FEATURE 6: sentence_length_variance ──────────────────────────
    # Standard deviation of sentence word counts
    # High = writer mixes short and long sentences
    # Low  = writer is very consistent in sentence length
    mean   = avg_sentence_length
    variance = sum((c - mean) ** 2 for c in sentence_word_counts) / len(sentence_word_counts)
    sentence_length_variance = variance ** 0.5  # square root = std deviation

    # ── FEATURE 7: comma_frequency ───────────────────────────────────
    comma_count      = text.count(',')
    comma_frequency  = comma_count / len(sentences)

    # ── FEATURE 8: question_mark_frequency ───────────────────────────
    # Per 100 sentences so short texts aren't unfairly penalised
    question_count            = text.count('?')
    question_mark_frequency   = (question_count / len(sentences)) * 100

    # ── FEATURE 9: exclamation_frequency ─────────────────────────────
    exclamation_count         = text.count('!')
    exclamation_frequency     = (exclamation_count / len(sentences)) * 100

    # ── FEATURE 10: paragraph_length ─────────────────────────────────
    # Split on blank lines to find paragraphs
    # Then average how many sentences each paragraph contains
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paragraphs) == 0:
        paragraphs = [text]  # treat whole text as one paragraph

    para_sentence_counts = []
    for para in paragraphs:
        para_sents = nltk.sent_tokenize(para)
        para_sentence_counts.append(len(para_sents))

    paragraph_length = sum(para_sentence_counts) / len(para_sentence_counts)

    # ── FEATURE 11: lexical_density ──────────────────────────────────
    # POS tag every word, then count content words:
    # NN=noun, VB=verb, JJ=adjective, RB=adverb
    # We lowercase for tagging consistency
    tagged = nltk.pos_tag([w.lower() for w in alpha_words])
    content_tags = {'NN','NNS','NNP','NNPS',   # nouns
                    'VB','VBD','VBG','VBN','VBP','VBZ',  # verbs
                    'JJ','JJR','JJS',           # adjectives
                    'RB','RBR','RBS'}           # adverbs
    content_count   = sum(1 for _, tag in tagged if tag in content_tags)
    lexical_density = content_count / len(alpha_words)

    # ── RETURN ALL 11 FEATURES AS A DICT ─────────────────────────────
    return {
        'avg_sentence_length':      avg_sentence_length,
        'vocab_richness':           vocab_richness,
        'punctuation_density':      punctuation_density,
        'avg_word_length':          avg_word_length,
        'function_word_ratio':      function_word_ratio,
        'sentence_length_variance': sentence_length_variance,
        'comma_frequency':          comma_frequency,
        'question_mark_frequency':  question_mark_frequency,
        'exclamation_frequency':    exclamation_frequency,
        'paragraph_length':         paragraph_length,
        'lexical_density':          lexical_density,
    }