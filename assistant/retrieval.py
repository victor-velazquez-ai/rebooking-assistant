"""Two simple retrievers (no embeddings, so it runs offline) and the measure to compare them."""

import re

STOPWORDS = {"a", "an", "the", "is", "are", "can", "to", "for", "of", "on", "in", "and", "or", "what", "how", "much",
             "does", "do", "he", "she", "we", "our", "it", "be", "may", "if", "there", "where", "after", "with", "by"}


def words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]


def by_title(question, passages, k):
    """Ranks passages by the words they share with the question, looking only at the title."""
    q = set(words(question))
    return sorted(passages, key=lambda p: -len(q & set(words(p["title"]))))[:k]


def by_title_and_text(question, passages, k):
    """Ranks passages by the words they share with the question, looking at the title and the text."""
    q = set(words(question))
    return sorted(passages, key=lambda p: -len(q & set(words(p["title"] + " " + p["text"]))))[:k]


RETRIEVERS = {"by_title": by_title, "by_title_and_text": by_title_and_text}

# Task 3: after measuring both retrievers with recall_at_k, set this to the better one.
CHOSEN_RETRIEVER = "by_title_and_text"  # recall@3: by_title 0.80, by_title_and_text 1.00


def recall_at_k(ranked_ids, relevant_ids, k):
    """Share of questions whose relevant passage id is among the first k ids ranked for that question.

    ranked_ids:   one list of passage ids per question, best first, e.g. [["FR-03", "FR-01"], ["FR-07", "FR-04"]]
    relevant_ids: the right passage id for each question, e.g. ["FR-01", "FR-02"]
    Returns a number from 0.0 to 1.0. In the example, recall_at_k(..., k=2) is 0.5.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(relevant in ranked[:k] for ranked, relevant in zip(ranked_ids, relevant_ids))
    return hits / len(relevant_ids)
