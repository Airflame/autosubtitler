"""Language tables shared by the transcriber (Whisper) and translator (NLLB-200).

Keys are the language codes Whisper uses (mostly ISO 639-1). Each entry holds a
human-readable name and the matching NLLB-200 code, or None when NLLB has no
equivalent (such sources are routed through Whisper's built-in English
translation first).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str          # Whisper code
    name: str
    nllb: str | None   # NLLB-200 code
    iso: str           # code used in output file names (e.g. movie.pl.srt)
    spaced: bool = True  # False for scripts written without spaces between words


_RAW: list[tuple] = [
    # code, name, nllb, [iso], [spaced]
    ("af", "Afrikaans", "afr_Latn"),
    ("am", "Amharic", "amh_Ethi"),
    ("ar", "Arabic", "arb_Arab"),
    ("as", "Assamese", "asm_Beng"),
    ("az", "Azerbaijani", "azj_Latn"),
    ("ba", "Bashkir", "bak_Cyrl"),
    ("be", "Belarusian", "bel_Cyrl"),
    ("bg", "Bulgarian", "bul_Cyrl"),
    ("bn", "Bengali", "ben_Beng"),
    ("bo", "Tibetan", "bod_Tibt", "bo", False),
    ("br", "Breton", None),
    ("bs", "Bosnian", "bos_Latn"),
    ("ca", "Catalan", "cat_Latn"),
    ("cs", "Czech", "ces_Latn"),
    ("cy", "Welsh", "cym_Latn"),
    ("da", "Danish", "dan_Latn"),
    ("de", "German", "deu_Latn"),
    ("el", "Greek", "ell_Grek"),
    ("en", "English", "eng_Latn"),
    ("es", "Spanish", "spa_Latn"),
    ("et", "Estonian", "est_Latn"),
    ("eu", "Basque", "eus_Latn"),
    ("fa", "Persian", "pes_Arab"),
    ("fi", "Finnish", "fin_Latn"),
    ("fo", "Faroese", "fao_Latn"),
    ("fr", "French", "fra_Latn"),
    ("gl", "Galician", "glg_Latn"),
    ("gu", "Gujarati", "guj_Gujr"),
    ("ha", "Hausa", "hau_Latn"),
    ("haw", "Hawaiian", None),
    ("he", "Hebrew", "heb_Hebr"),
    ("hi", "Hindi", "hin_Deva"),
    ("hr", "Croatian", "hrv_Latn"),
    ("ht", "Haitian Creole", "hat_Latn"),
    ("hu", "Hungarian", "hun_Latn"),
    ("hy", "Armenian", "hye_Armn"),
    ("id", "Indonesian", "ind_Latn"),
    ("is", "Icelandic", "isl_Latn"),
    ("it", "Italian", "ita_Latn"),
    ("ja", "Japanese", "jpn_Jpan", "ja", False),
    ("jw", "Javanese", "jav_Latn", "jv"),
    ("ka", "Georgian", "kat_Geor"),
    ("kk", "Kazakh", "kaz_Cyrl"),
    ("km", "Khmer", "khm_Khmr", "km", False),
    ("kn", "Kannada", "kan_Knda"),
    ("ko", "Korean", "kor_Hang"),
    ("la", "Latin", None),
    ("lb", "Luxembourgish", "ltz_Latn"),
    ("ln", "Lingala", "lin_Latn"),
    ("lo", "Lao", "lao_Laoo", "lo", False),
    ("lt", "Lithuanian", "lit_Latn"),
    ("lv", "Latvian", "lvs_Latn"),
    ("mg", "Malagasy", "plt_Latn"),
    ("mi", "Maori", "mri_Latn"),
    ("mk", "Macedonian", "mkd_Cyrl"),
    ("ml", "Malayalam", "mal_Mlym"),
    ("mn", "Mongolian", "khk_Cyrl"),
    ("mr", "Marathi", "mar_Deva"),
    ("ms", "Malay", "zsm_Latn"),
    ("mt", "Maltese", "mlt_Latn"),
    ("my", "Burmese", "mya_Mymr", "my", False),
    ("ne", "Nepali", "npi_Deva"),
    ("nl", "Dutch", "nld_Latn"),
    ("nn", "Norwegian Nynorsk", "nno_Latn"),
    ("no", "Norwegian", "nob_Latn"),
    ("oc", "Occitan", "oci_Latn"),
    ("pa", "Punjabi", "pan_Guru"),
    ("pl", "Polish", "pol_Latn"),
    ("ps", "Pashto", "pbt_Arab"),
    ("pt", "Portuguese", "por_Latn"),
    ("ro", "Romanian", "ron_Latn"),
    ("ru", "Russian", "rus_Cyrl"),
    ("sa", "Sanskrit", "san_Deva"),
    ("sd", "Sindhi", "snd_Arab"),
    ("si", "Sinhala", "sin_Sinh"),
    ("sk", "Slovak", "slk_Latn"),
    ("sl", "Slovenian", "slv_Latn"),
    ("sn", "Shona", "sna_Latn"),
    ("so", "Somali", "som_Latn"),
    ("sq", "Albanian", "als_Latn"),
    ("sr", "Serbian", "srp_Cyrl"),
    ("su", "Sundanese", "sun_Latn"),
    ("sv", "Swedish", "swe_Latn"),
    ("sw", "Swahili", "swh_Latn"),
    ("ta", "Tamil", "tam_Taml"),
    ("te", "Telugu", "tel_Telu"),
    ("tg", "Tajik", "tgk_Cyrl"),
    ("th", "Thai", "tha_Thai", "th", False),
    ("tk", "Turkmen", "tuk_Latn"),
    ("tl", "Tagalog", "tgl_Latn"),
    ("tr", "Turkish", "tur_Latn"),
    ("tt", "Tatar", "tat_Cyrl"),
    ("uk", "Ukrainian", "ukr_Cyrl"),
    ("ur", "Urdu", "urd_Arab"),
    ("uz", "Uzbek", "uzn_Latn"),
    ("vi", "Vietnamese", "vie_Latn"),
    ("yi", "Yiddish", "ydd_Hebr"),
    ("yo", "Yoruba", "yor_Latn"),
    ("yue", "Cantonese", "yue_Hant", "yue", False),
    ("zh", "Chinese", "zho_Hans", "zh", False),
]


def _build() -> dict[str, Language]:
    table = {}
    for row in _RAW:
        code, name, nllb = row[:3]
        iso = row[3] if len(row) > 3 else code
        spaced = row[4] if len(row) > 4 else True
        table[code] = Language(code, name, nllb, iso, spaced)
    return table


LANGUAGES: dict[str, Language] = _build()


def get(code: str) -> Language:
    """Return the language for a Whisper code, or a minimal placeholder if unknown."""
    return LANGUAGES.get(code) or Language(code, code, None, code)


def source_languages() -> list[Language]:
    """Languages Whisper can transcribe, sorted by name."""
    return sorted(LANGUAGES.values(), key=lambda l: l.name)


def target_languages() -> list[Language]:
    """Languages NLLB can translate into, sorted by name."""
    return [l for l in source_languages() if l.nllb]
