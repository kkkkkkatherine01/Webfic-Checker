"""Chinese word segmentation for keyword search (jieba).

Character names and aliases are added to the dictionary first, so "旗木卡卡西" stays one
token instead of being cut into pieces that also match other text.
"""

import logging
import re
from collections.abc import Iterable
from pathlib import Path

# Tokens worth matching: words with at least one letter, digit or CJK character.
_WORD = re.compile(r"[\w一-鿿]")
# Very common function words that would make every passage match.
_STOP = frozenset(
    [
        "的",
        "了",
        "着",
        "是",
        "在",
        "和",
        "与",
        "也",
        "就",
        "都",
        "而",
        "及",
        "或",
        "一个",
        "这",
        "那",
        "他",
        "她",
        "它",
        "我",
        "你",
        "们",
        "吗",
        "呢",
        "吧",
        "啊",
    ]
)


class Tokenizer:
    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir
        self._jieba = None
        self._known: set[str] = set()

    def _load(self):
        if self._jieba is None:
            import jieba  # loads its dictionary on first use

            jieba.setLogLevel(logging.WARNING)
            tokenizer = jieba.Tokenizer()
            if self._cache_dir:
                folder = self._cache_dir / "jieba"
                folder.mkdir(parents=True, exist_ok=True)
                tokenizer.tmp_dir = str(folder)
            self._jieba = tokenizer
        return self._jieba

    def add_names(self, names: Iterable[str]) -> None:
        jieba = self._load()
        for name in names:
            name = name.strip()
            if len(name) >= 2 and name not in self._known:
                jieba.add_word(name, freq=100_000)
                self._known.add(name)

    def tokens(self, text: str) -> list[str]:
        words = (w.strip() for w in self._load().lcut(text))
        return [w for w in words if w and w not in _STOP and _WORD.search(w)]
