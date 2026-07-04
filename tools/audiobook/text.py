"""Clean ebook markdown and break it into small, speakable parts.

The functions here are pure (text in, data out) and hold no I/O, so they can be
unit-tested in isolation.
"""

import re

MAX_CHARS = 200

# Sentence terminators (incl. closing quotes/paren that may trail them).
# Alternation of fixed-width lookbehinds: Python forbids a variable-width one.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’\)\]]))\s+")
# Clause boundaries used to break a sentence that is longer than MAX_CHARS.
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—–])\s+")


def clean_markdown(text: str) -> str:
    """Strip common markdown syntax so only spoken text remains."""
    # Code fences and inline code.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Images and links -> keep the visible text.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Heading markers, blockquotes, list bullets at line start.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", text)
    # Horizontal rules.
    text = re.sub(r"(?m)^\s*([-*_])\1{2,}\s*$", "", text)
    # Emphasis markers (leave the words, drop the surrounding * or _).
    text = re.sub(r"(\*\*|__|\*|_)(.+?)\1", r"\2", text)
    return text


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Split a too-long sentence into <= max_chars pieces.

    Tries clause boundaries first, then falls back to splitting on spaces so no
    piece ever exceeds the limit.
    """
    pieces: list[str] = []
    for clause in _CLAUSE_SPLIT.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_chars:
            pieces.append(clause)
            continue
        # Still too long: wrap on word boundaries.
        current = ""
        for word in clause.split():
            if len(word) > max_chars:
                # A single monstrous token; chop it by characters.
                if current:
                    pieces.append(current)
                    current = ""
                for i in range(0, len(word), max_chars):
                    pieces.append(word[i:i + max_chars])
                continue
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= max_chars:
                current += " " + word
            else:
                pieces.append(current)
                current = word
        if current:
            pieces.append(current)
    return pieces


def build_parts(markdown: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Turn cleaned markdown into an ordered list of parts.

    Each returned dict has: part_id, text, chars.
    """
    # Split into blank-line-separated blocks first.
    raw_blocks = re.split(r"\n\s*\n", markdown)

    blocks: list[str] = []
    for block in raw_blocks:
        block = clean_markdown(block)
        # Join wrapped lines and collapse whitespace within the block.
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        # PDF-extracted text sometimes breaks a single sentence across blank
        # lines; if the previous block did not end a sentence, glue this one on.
        if blocks and not re.search(r"[.!?…][\"'”’\)\]]?$", blocks[-1]):
            blocks[-1] = (blocks[-1] + " " + block).strip()
        else:
            blocks.append(block)

    # Break each block into sentences, then greedily pack them into chunks.
    chunks: list[str] = []
    for block in blocks:
        current = ""
        for sentence in _SENTENCE_END.split(block):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(_hard_split(sentence, max_chars))
                continue
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= max_chars:
                current += " " + sentence
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

    parts = []
    for i, text in enumerate(chunks, start=1):
        # The TTS text must contain no periods; replace every "." with ",".
        # Done after splitting so sentence boundaries are still detected above.
        text = text.replace(".", ",")
        parts.append({"part_id": i, "text": text, "chars": len(text)})
    return parts
