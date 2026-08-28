SYSTEM = """You rank a list of numbered passages by how relevant each one \
is to answering a question. You do NOT answer the question.

Rules:
- Output ONLY a JSON array of the passage numbers, most relevant first
  (e.g. [3, 0, 5]). No prose, no explanation, no markdown fences.
- Include a passage only if it is actually useful for answering the
  question - omit passages that are unrelated, even if they mention
  similar words. It is fine to return fewer passages than were given, or
  an empty array [] if none of them are actually relevant.
- Two passages can share a sentence at their boundary (they're adjacent
  pieces of the same longer document, split apart) while each still
  containing different relevant content beyond that shared sentence -
  don't drop one just because it looks like it repeats another; judge
  each passage's full content, not just its opening/closing line.
- Never include a passage number that wasn't in the input."""


def build_user_prompt(*, question: str, candidates: list[str], top_n: int) -> str:
    passages = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(candidates))
    return (
        f"Question: {question}\n\n"
        f"Passages:\n{passages}\n\n"
        f"Return the passage numbers of at most the {top_n} most relevant passages, "
        "most relevant first, as a JSON array:"
    )
