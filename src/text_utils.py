"""Nettoyage des réponses du modèle."""
import re


def clean(text: str) -> str:
    plain = re.sub(r"<[^>]+>", "", text).strip()
    parts = re.split(r"([.!?](?:\s+|$)|\n+)", plain)
    seen, result = set(), []
    for index in range(0, len(parts), 2):
        sentence = parts[index].strip()
        if not sentence:
            continue
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        key = re.sub(r"\s+", " ", sentence).casefold()
        if key not in seen:
            seen.add(key)
            result.append((sentence + separator).strip())
    return " ".join(result)
