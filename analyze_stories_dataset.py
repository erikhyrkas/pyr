from datasets import load_dataset
import re
from collections import defaultdict
import matplotlib.pyplot as plt

dataset = load_dataset("OmniAICreator/RoyalRoad-1.61M", split="train")

# Possible warnings: Profanity, Sexual Content, Disturbing Content, and Graphic Violence
EXCLUDE_WARNINGS = {
    "Sexual Content",
    "Disturbing Content"
}
EXCLUDE_GENRES = {
    "Grimdark",
    "Harem"
}
offensive_patterns = [
    r"\bnigger\b",
    r"\bfaggot\b",
    r"\blolita\b",
    r"\bloli\b",
]
compiled_patterns = [re.compile(p, re.IGNORECASE) for p in offensive_patterns]

author_stats = defaultdict(lambda: {"total": 0, "bad": 0, "skipped": 0})
MIN_VIEWS = 300
for i, entry in enumerate(dataset):
    author = entry["author"]
    text = entry["text"]
    average_views = entry["average_views"]
    tags = entry.get("tags", [])
    warnings = entry.get("warnings", [])
    should_skip = False
    if author is None or text is None or any(tag in EXCLUDE_GENRES for tag in tags) or any(warning in EXCLUDE_WARNINGS for warning in warnings) or average_views < MIN_VIEWS:
        should_skip = True
    elif any(p.search(text) for p in compiled_patterns):
        author_stats[author]["bad"] += 1

    if should_skip:
        author_stats[author]["skipped"] += 1
    else:
        author_stats[author]["total"] += 1


    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1:,} chapters...")

ratios = []
for author, stats in author_stats.items():
    if stats["total"] >= 3:
        ratio = stats["bad"] / stats["total"]
        ratios.append(ratio)

plt.figure(figsize=(10, 6))
plt.hist(ratios, bins=40, color='orchid', edgecolor='black')
plt.title("Infraction Ratios per Author (≥3 chapters)")
plt.xlabel("Proportion of Flagged Chapters")
plt.ylabel("Number of Authors")
plt.grid(True)
plt.tight_layout()
plt.savefig("author_infraction_histogram.png")
plt.show()

# Save flagged authors to file
flagged_authors = [
    author for author, stats in author_stats.items()
    if (stats["total"] < 3) or stats["bad"] >= 2 or (stats["total"] >= 3 and stats["bad"] / stats["total"] > 0.1)
]

with open("excluded_authors.txt", "w", encoding="utf-8") as f:
    for author in flagged_authors:
        f.write(author + "\n")

print(f"\nSaved {len(flagged_authors)} flagged authors to excluded_authors.txt")

print("\nTop flagged authors:")
top = sorted(
    ((author, s["bad"], s["total"], s["bad"] / s["total"])
     for author, s in author_stats.items() if s["bad"] > 0),
    key=lambda x: (-x[1], x[0])
)

for author, bad, total, ratio in top[:20]:
    print(f"{author:30} | {bad:>3}/{total:<3} ({ratio:.1%})")
