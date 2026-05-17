import re
import time
import requests





AUTHOR_BOOKS = {
    'Jane Austen':        [1342, 161, 121],
    'Charles Dickens':    [98,   1400, 730],
    'Mark Twain':         [74,   76,   86],
    'Oscar Wilde':        [174,  854,  333],
    'Virginia Woolf':     [5670, 144,  4476],
    'Edgar Allan Poe':    [2147, 932,  25525],
    'Arthur Conan Doyle': [1661, 2097, 108],
}

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CHUNK_SIZE   = 500   # words per training chunk
SLEEP_DELAY  = 2.0   # seconds between Gutenberg requests — do not lower this
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def fetch_raw_text(book_id: int) -> str | None:
    """
    Downloads a single book from Gutenberg by its numeric ID.
    Returns the raw text string, or None if the request fails.

    Common mistake: Gutenberg sometimes uses a mirror URL format.
    If this URL pattern fails for a specific book, the fallback
    URL pattern is: https://www.gutenberg.org/files/{id}/{id}-0.txt
    """
    url = GUTENBERG_URL.format(id=book_id)
    try:
        # stream=False is fine — books are only a few MB at most
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # raises exception for 4xx/5xx status codes

        # Gutenberg files are utf-8 but sometimes have latin-1 fallback
        # This handles both without crashing
        return response.content.decode('utf-8', errors='replace')

    except requests.RequestException as e:
        print(f"  [!] Failed to fetch book {book_id}: {e}")
        return None


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Removes Gutenberg's standard header and footer from raw text.

    The header always ends with a line containing:
        *** START OF THE PROJECT GUTENBERG EBOOK ... ***
    The footer always starts with a line containing:
        *** END OF THE PROJECT GUTENBERG EBOOK ... ***

    We find these markers and slice out only the middle — the actual book.

    Common mistake: some older books use slightly different marker formats.
    The regex here handles both the old and new Gutenberg formats.
    """
    # This pattern matches both old and new Gutenberg header formats
    start_pattern = r'\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*{3}'
    end_pattern   = r'\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*{3}'

    start_match = re.search(start_pattern, text, re.IGNORECASE)
    end_match   = re.search(end_pattern,   text, re.IGNORECASE)

    if start_match and end_match:
        # Slice from end of header marker to start of footer marker
        cleaned = text[start_match.end() : end_match.start()]
    elif start_match:
        # Footer marker missing — just strip the header
        cleaned = text[start_match.end():]
    else:
        # No markers found at all — use the whole text
        # This happens occasionally with older Gutenberg formats
        print("  [!] Gutenberg markers not found — using full text")
        cleaned = text

    return cleaned.strip()


def clean_text(text: str) -> str:
    """
    Light cleaning of book text after boilerplate is removed.

    We do NOT aggressively clean — we want to preserve:
    - Punctuation  (needed for punctuation_density feature)
    - Sentence structure (needed for avg_sentence_length)
    - Paragraphs   (needed for paragraph_length feature)

    We only remove things that would corrupt our features:
    - Excessive blank lines (3+ in a row → 2)
    - Chapter headings in ALL CAPS (not author prose)
    - Page numbers / roman numerals on their own line
    """
    # Collapse 3+ blank lines into 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are purely uppercase (chapter headings)
    # e.g. "CHAPTER I" or "BOOK THE FIRST"
    lines = text.split('\n')
    lines = [
        line for line in lines
        if not (line.strip().isupper() and len(line.strip()) < 60)
    ]
    text = '\n'.join(lines)

    # Remove standalone Roman numerals / chapter numbers on their own line
    text = re.sub(r'^\s*(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\s*$',
                  '', text, flags=re.MULTILINE | re.IGNORECASE)

    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Splits cleaned text into chunks of approximately `chunk_size` words.

    Why chunk? One novel = one data point if we don't chunk.
    Chunking gives us ~150-200 data points per book.
    More data points = better trained model.

    We split on whitespace, group into chunks, then rejoin.
    We discard the final chunk if it's less than half the target size —
    a 50-word tail chunk would skew our features significantly.

    Common mistake: splitting by characters instead of words.
    Character splits can cut sentences mid-word, corrupting sentence tokenisation.
    """
    # Split entire text into individual word tokens
    words = text.split()

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]

        # Discard undersized tail chunks
        if len(chunk_words) < chunk_size // 2:
            break

        # Rejoin words back into a string — this is what extract_features() receives
        chunks.append(' '.join(chunk_words))

    return chunks


def fetch_author_chunks(author: str, book_ids: list[int]) -> list[str]:
    """
    For a given author, fetches all their books, cleans them,
    and returns a flat list of ~500-word text chunks.

    This is the main function called by train.py.

    Returns a list of strings — each string is one training sample.
    """
    all_chunks = []

    print(f"\n[{author}]")

    for book_id in book_ids:
        print(f"  Fetching book ID {book_id}...", end=' ')

        raw = fetch_raw_text(book_id)
        if raw is None:
            continue  # skip this book, try the next one

        # Strip Gutenberg boilerplate
        stripped = strip_gutenberg_boilerplate(raw)

        # Light cleaning
        cleaned = clean_text(stripped)

        # Chunk into ~500 word blocks
        chunks = chunk_text(cleaned)

        print(f"got {len(chunks)} chunks")
        all_chunks.extend(chunks)

        # Be polite to Gutenberg's servers
        # Common mistake: forgetting this — your IP will get rate limited
        time.sleep(SLEEP_DELAY)

    print(f"  Total chunks for {author}: {len(all_chunks)}")
    return all_chunks


def fetch_all_authors() -> dict[str, list[str]]:
    """
    Fetches chunks for every author in AUTHOR_BOOKS.
    Returns a dict: { author_name: [chunk1, chunk2, ...] }

    This is what train.py calls once to get all training data.
    """
    all_data = {}

    for author, book_ids in AUTHOR_BOOKS.items():
        chunks = fetch_author_chunks(author, book_ids)
        if chunks:
            all_data[author] = chunks
        else:
            print(f"  [!] No chunks retrieved for {author} — skipping")

    return all_data


# ── QUICK TEST ────────────────────────────────────────────────────────────────
# Run this file directly to test a single book fetch
# python gutenberg.py
if __name__ == '__main__':
    print("Testing gutenberg.py — fetching one Austen book...\n")

    raw     = fetch_raw_text(1342)          # Pride and Prejudice
    stripped = strip_gutenberg_boilerplate(raw)
    cleaned  = clean_text(stripped)
    chunks   = chunk_text(cleaned)

    print(f"Raw text length:     {len(raw):,} characters")
    print(f"Cleaned text length: {len(cleaned):,} characters")
    print(f"Number of chunks:    {len(chunks)}")
    print(f"\nFirst 200 chars of chunk 1:\n{chunks[0][:200]}")
    print(f"\nFirst 200 chars of chunk 2:\n{chunks[1][:200]}")