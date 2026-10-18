"""Turn Whisper output into readable subtitle cues.

Whisper segments are often too long (or too short) to read comfortably, so cues
are rebuilt from word timestamps using common subtitling rules: a limited number
of characters per line, at most two lines, a maximum on-screen duration, and
breaks at sentence ends and pauses.
"""

from __future__ import annotations

from dataclasses import dataclass

from .srt import Cue

SENTENCE_END = (".", "?", "!", "…", "。", "？", "！", "।", "؟")
CLAUSE_END = (",", ";", ":", "，", "、", "；", "：")


@dataclass
class Word:
    start: float
    end: float
    text: str  # includes the leading space for space-separated languages


@dataclass
class SegmentOptions:
    max_line_chars: int = 42
    max_lines: int = 2
    max_duration: float = 6.0
    min_duration: float = 1.0
    pause_split: float = 0.8  # start a new cue after a silence this long
    linger: float = 0.4       # keep text on screen a bit after the last word


def join_words(words: list[Word]) -> str:
    return "".join(w.text for w in words).strip()


def build_cues(words: list[Word], opts: SegmentOptions, spaced: bool = True) -> list[Cue]:
    max_chars = opts.max_line_chars * opts.max_lines
    groups: list[list[Word]] = []
    current: list[Word] = []

    for word in words:
        if current:
            gap = word.start - current[-1].end
            candidate = join_words(current + [word])
            too_long = len(candidate) > max_chars
            too_slow = word.end - current[0].start > opts.max_duration
            if gap >= opts.pause_split or too_long or too_slow:
                groups.extend(_split_at_clause(current, max_chars) if too_long else [current])
                current = []
        current.append(word)
        if word.text.strip().endswith(SENTENCE_END):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    cues = [Cue(g[0].start, g[-1].end, join_words(g)) for g in groups if join_words(g)]
    _fix_timing(cues, opts)
    for cue in cues:
        cue.text = wrap(cue.text, opts.max_line_chars, spaced)
    return cues


def _split_at_clause(words: list[Word], max_chars: int) -> list[list[Word]]:
    """If a full cue has a comma in its second half, break there so the next cue
    starts at a natural clause boundary instead of mid-phrase."""
    for i in range(len(words) - 2, len(words) // 2 - 1, -1):
        if words[i].text.strip().endswith(CLAUSE_END):
            return [words[: i + 1], words[i + 1:]]
    return [words]


def _fix_timing(cues: list[Cue], opts: SegmentOptions) -> None:
    for i, cue in enumerate(cues):
        limit = cues[i + 1].start - 0.05 if i + 1 < len(cues) else float("inf")
        cue.end = min(max(cue.end + opts.linger, cue.start + opts.min_duration), limit)
        cue.end = max(cue.end, cue.start + 0.1)


def wrap(text: str, max_line_chars: int, spaced: bool = True) -> str:
    """Wrap text into at most two balanced lines (more only for very long text)."""
    text = " ".join(text.split()) if spaced else text.strip()
    if len(text) <= max_line_chars:
        return text
    if not spaced:
        n_lines = -(-len(text) // max_line_chars)
        size = -(-len(text) // n_lines)
        return "\n".join(text[i:i + size] for i in range(0, len(text), size))

    n_lines = max(2, -(-len(text) // (max_line_chars + 8)))
    words = text.split(" ")
    if len(words) == 1:
        return text
    target = len(text) / n_lines
    lines, line = [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and len(lines) < n_lines - 1 and len(candidate) > target and \
                abs(len(line) - target) <= abs(len(candidate) - target):
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    return "\n".join(lines)


def distribute(text: str, weights: list[int], spaced: bool = True) -> list[str]:
    """Split a translated sentence back across several cues, proportionally to the
    lengths of the source cues, breaking at word boundaries where possible."""
    if len(weights) == 1:
        return [text]
    units = text.split() if spaced else list(text.strip())
    if not units:
        return [""] * len(weights)
    sep = " " if spaced else ""
    total_weight = sum(weights) or len(weights)
    total_len = sum(len(u) for u in units) + (len(units) - 1) * len(sep)

    parts: list[str] = []
    idx = 0
    consumed = 0.0
    cumulative = 0
    for k, w in enumerate(weights):
        cumulative += w
        if k == len(weights) - 1:
            parts.append(sep.join(units[idx:]))
            break
        boundary = total_len * cumulative / total_weight
        chunk: list[str] = []
        # Leave at least one unit for each remaining cue, when there are enough units.
        reserve = len(weights) - k - 1 if len(units) >= len(weights) else 0
        while idx < len(units) - reserve:
            unit_len = len(units[idx]) + len(sep)
            if chunk and consumed + unit_len / 2 > boundary:
                break
            chunk.append(units[idx])
            consumed += unit_len
            idx += 1
        parts.append(sep.join(chunk))
    return parts


def sentence_groups(cues: list[Cue], max_cues: int = 4, max_chars: int = 400) -> list[list[int]]:
    """Group consecutive cue indices that together form one sentence, so they can be
    translated with full context."""
    groups: list[list[int]] = []
    current: list[int] = []
    length = 0
    for i, cue in enumerate(cues):
        current.append(i)
        length += len(cue.text)
        ends = cue.text.strip().endswith(SENTENCE_END)
        if ends or len(current) >= max_cues or length >= max_chars:
            groups.append(current)
            current, length = [], 0
    if current:
        groups.append(current)
    return groups
