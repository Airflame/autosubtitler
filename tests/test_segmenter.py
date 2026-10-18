from autosubtitler.segmenter import SegmentOptions, Word, build_cues, distribute, sentence_groups, wrap
from autosubtitler.srt import Cue, format_timestamp, to_srt


def words(text: str, start: float = 0.0, step: float = 0.3) -> list[Word]:
    out = []
    for i, w in enumerate(text.split()):
        t = start + i * step
        out.append(Word(t, t + step * 0.9, " " + w))
    return out


def test_timestamp_format():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3725.5) == "01:02:05,500"


def test_srt_numbering_skips_empty():
    srt = to_srt([Cue(0, 1, "a"), Cue(1, 2, "  "), Cue(2, 3, "b")])
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000\na\n")
    assert "\n2\n00:00:02,000 --> 00:00:03,000\nb\n" in srt


def test_sentences_become_separate_cues():
    cues = build_cues(words("Hello there. How are you?"), SegmentOptions())
    assert [c.text for c in cues] == ["Hello there.", "How are you?"]


def test_pause_splits_cue():
    w = words("one two") + words("three four", start=5.0)
    assert [c.text for c in build_cues(w, SegmentOptions())] == ["one two", "three four"]


def test_long_text_respects_limits():
    opts = SegmentOptions(max_line_chars=20, max_lines=2)
    text = " ".join(["word"] * 60)
    cues = build_cues(words(text, step=0.05), opts)
    for c in cues:
        assert len(c.text.replace("\n", " ")) <= 40
        assert c.text.count("\n") <= 1
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start


def test_wrap_balances_two_lines():
    lines = wrap("The quick brown fox jumps over the lazy dog again", 30).split("\n")
    assert len(lines) == 2
    assert abs(len(lines[0]) - len(lines[1])) < 12


def test_distribute_keeps_all_words_in_order():
    text = "Ala ma kota a kot ma Ale i wszyscy sa szczesliwi"
    parts = distribute(text, [10, 30, 10])
    assert len(parts) == 3 and all(parts)
    assert " ".join(parts) == text


def test_distribute_unspaced():
    parts = distribute("今日はとても良い天気ですね", [5, 5], spaced=False)
    assert "".join(parts) == "今日はとても良い天気ですね"
    assert all(parts)


def test_distribute_fewer_words_than_cues_fills_first():
    assert distribute("Hello", [5, 5]) == ["Hello", ""]


def test_sentence_groups():
    cues = [Cue(0, 1, "I went"), Cue(1, 2, "to the shop."), Cue(2, 3, "Then home.")]
    assert sentence_groups(cues) == [[0, 1], [2]]
