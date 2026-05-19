# test_data.py — run this before anything else
from gutenberg import fetch_all_authors

data = fetch_all_authors()
print("\n=== CHUNK COUNTS ===")
for author, chunks in data.items():
    total_words = sum(len(c.split()) for c in chunks)
    print(f"{author:25s}: {len(chunks):4d} chunks  (~{total_words:,} words)")
    print(f"  Sample: {chunks[0][:120]}")
    print()