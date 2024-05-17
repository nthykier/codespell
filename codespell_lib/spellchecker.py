#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.html.
"""
Copyright (C) 2010-2011  Lucas De Marchi <lucas.de.marchi@gmail.com>
Copyright (C) 2011  ProFUSION embedded systems
"""

import re
import os
from typing import (
    Dict,
    Sequence,
    Container,
    Optional,
    Iterable,
    Protocol,
    Generic,
    TypeVar,
)

# Pass all misspellings through this translation table to generate
# alternative misspellings and fixes.
alt_chars = (("'", "’"),)  # noqa: RUF001

T_co = TypeVar("T_co", bound="Token", covariant=True)


supported_languages_en = ("en", "en_GB", "en_US", "en_CA", "en_AU")
supported_languages = supported_languages_en

# Users might want to link this file into /usr/local/bin, so we resolve the
# symbolic link path to the real path if necessary.
_data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
_builtin_dictionaries = (
    # name, desc, name, err in aspell, correction in aspell, \
    # err dictionary array, rep dictionary array
    # The arrays must contain the names of aspell dictionaries
    # The aspell tests here aren't the ideal state, but the None's are
    # realistic for obscure words
    ("clear", "for unambiguous errors", "", False, None, supported_languages_en, None),
    (
        "rare",
        "for rare (but valid) words that are likely to be errors",
        "_rare",
        None,
        None,
        None,
        None,
    ),
    (
        "informal",
        "for making informal words more formal",
        "_informal",
        True,
        True,
        supported_languages_en,
        supported_languages_en,
    ),
    (
        "usage",
        "for replacing phrasing with recommended terms",
        "_usage",
        None,
        None,
        None,
        None,
    ),
    (
        "code",
        "for words from code and/or mathematics that are likely to be typos in other contexts (such as uint)",  # noqa: E501
        "_code",
        None,
        None,
        None,
        None,
    ),
    (
        "names",
        "for valid proper names that might be typos",
        "_names",
        None,
        None,
        None,
        None,
    ),
    (
        "en-GB_to_en-US",
        "for corrections from en-GB to en-US",
        "_en-GB_to_en-US",
        True,
        True,
        ("en_GB",),
        ("en_US",),
    ),
)
_builtin_default = "clear,rare"

_builtin_default_as_tuple = tuple(_builtin_default.split(","))


class UnknownBuiltinDictionaryError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown built-in dictionary: {name}")


class BuiltinDictionariesAlreadyLoadedError(TypeError):
    def __init__(self) -> None:
        super().__init__(
            "load_builtin_dictionaries must not be called more than once",
        )


class LineTokenizer(Protocol[T_co]):
    """Callable that splits a line into multiple tokens to be spellchecked

    Generally, a regex will do for simple cases. A probably too simple one is:

        >>> tokenizer = re.compile(r"[^ ]+").finditer

    For more complex cases, either use more complex regexes or custom tokenization
    code.
    """

    def __call__(self, line: str) -> Iterable[T_co]: ...


class Token(Protocol):
    """Describes a token

    This is a protocol to support `re.Match[str]` (which codespell uses) and any
    other tokenization method that our API consumers might be using.
    """

    def group(self) -> str: ...

    def start(self) -> int: ...


class Misspelling:
    def __init__(self, candidates: Sequence[str], fix: bool, reason: str) -> None:
        self.candidates = candidates
        self.fix = fix
        self.reason = reason


class DetectedMisspelling(Generic[T_co]):
    def __init__(
        self,
        word: str,
        lword: str,
        misspelling: Misspelling,
        token: T_co,
    ) -> None:
        self.word = word
        self.lword = lword
        self.misspelling = misspelling
        self.token = token


class Spellchecker:
    """The spellchecking dictionaries of codespell

    The Spellchecker is responsible for spellchecking words or lines. It maintains state
    for known typos, their corrections and known ignored words.

        >>> import re
        >>> s = Spellchecker()
        >>> # Very simple tokenizer
        >>> tokenizer = re.compile(r"[^ ]+").finditer
        >>> line = "A touple tpyo but also correct words appear" # codespell:ignore
        >>> issues = list(s.spellcheck_line(line, tokenizer))
        >>> len(issues) == 2
        >>> issues[0].word
        'touple'
        >>> list(issues[0].misspelling.candidates)
        ['tuple', 'couple', 'topple', 'toupee']
        >>> issues[0].misspelling.fix
        False
        >>> issues[1].word
        'tpyo'
        >>> list(issues[1].misspelling.candidates)
        ['typo']
        >>> issues[1].misspelling.fix
        True
    """

    def __init__(
        self,
        *,
        builtin_dictionaries: Optional[Sequence[str]] = _builtin_default_as_tuple,
    ) -> None:
        self._misspellings: Dict[str, Misspelling] = {}
        self._builtin_loaded = False
        self.ignore_words_cased: Container[str] = frozenset()
        if builtin_dictionaries:
            self.load_builtin_dictionaries(builtin_dictionaries)

    def spellcheck_line(
        self,
        line: str,
        tokenizer: LineTokenizer[T_co],
        *,
        extra_words_to_ignore: Container[str] = frozenset()
    ) -> Iterable[DetectedMisspelling[T_co]]:
        """Tokenize and spellcheck a line

        Split the line into tokens based using the provided tokenizer. See the doc
        string for the class for an example.

        :param line: The line to spellcheck.
        :param tokenizer: A callable that will tokenize the line
        :param extra_words_to_ignore: Extra words to ignore for this particular line
          (such as content from a `codespell:ignore` comment)
        """
        misspellings = self._misspellings
        ignore_words_cased = self.ignore_words_cased

        for token in tokenizer(line):
            word = token.group()
            if word in ignore_words_cased:
                continue
            lword = word.lower()
            misspelling = misspellings.get(lword)
            if misspelling is not None and lword not in extra_words_to_ignore:
                # Sometimes we find a 'misspelling' which is actually a valid word
                # preceded by a string escape sequence.  Ignore such cases as
                # they're usually false alarms; see issue #17 among others.
                char_before_idx = token.start() - 1
                if (
                    char_before_idx >= 0
                    and line[char_before_idx] == "\\"
                    # bell, backspace, formfeed, newline, carriage-return, tab, vtab.
                    and word.startswith(("a", "b", "f", "n", "r", "t", "v"))
                    and lword[1:] not in misspellings
                ):
                    continue
                yield DetectedMisspelling(word, lword, misspelling, token)

    def check_lower_cased_word(self, word: str) -> Optional[Misspelling]:
        """Check a given word against the loaded dictionaries

        :param word: The word to check. This should be all lower-case.
        """
        return self._misspellings.get(word)

    def load_builtin_dictionaries(
        self,
        builtin_dictionaries: Iterable[str] = _builtin_default_as_tuple,
        *,
        ignore_words: Container[str] = frozenset(),
    ) -> None:
        """Load codespell builtin dictionaries (for manual dictionary load order)

        This method enables you to load builtin dictionaries in a special order relative
        to custom dictionaries. To use this method, you must ensure that the constructor
        did *not* load any builtin dictionaries.

           >>> s = Spellchecker(builtin_dictionaries=None)
           >>> # A couple of s.load_dictionary_from_file(...) lines here
           >>> s.load_builtin_dictionaries("clear")

        This method updates the spellchecker to include any corrected listed
        in the file. Load order is important. When multiple corrections are
        loaded for the same typo, then the last loaded corrections for that
        typo will be used.

        :param builtin_dictionaries: Names of the codespell dictionaries to load
        :param ignore_words: Words to ignore from this dictionary.
        """
        if self._builtin_loaded:
            # It would work, but if you are doing manual load order, then probably
            # you will want to be sure it you get it correct.
            raise BuiltinDictionariesAlreadyLoadedError()
        for name in sorted(set(builtin_dictionaries)):
            self._load_builtin_dictionary(name, ignore_words=ignore_words)
        self._builtin_loaded = True

    def _load_builtin_dictionary(
        self,
        name: str,
        *,
        ignore_words: Container[str] = frozenset(),
    ) -> None:
        for builtin in _builtin_dictionaries:
            if builtin[0] == name:
                filename = os.path.join(_data_root, f"dictionary{builtin[2]}.txt")
                self.load_dictionary_from_file(filename, ignore_words=ignore_words)
                return
        raise UnknownBuiltinDictionaryError(name)

    def load_dictionary_from_file(
        self,
        filename: str,
        *,
        ignore_words: Container[str] = frozenset(),
    ) -> None:
        """Parse a codespell dictionary

        This is primarily useful for loading custom dictionaries not provided by
        codespell. This is the API version of the `-D` / `--dictionary` command
        line option except it only accept files (and not the special `-`).

        This method updates the spellchecker to include any corrected listed in
        the file. Load order is important. When multiple corrections are loaded
        for the same typo, then the last loaded corrections for that typo will
        be used.

        :param filename: The codespell dictionary file to parse
        :param ignore_words: Words to ignore from this dictionary.
        """
        misspellings = self._misspellings
        with open(filename, encoding="utf-8") as f:
            translate_tables = [(x, str.maketrans(x, y)) for x, y in alt_chars]
            for line in f:
                [key, data] = line.split("->")
                # TODO: For now, convert both to lower.
                #       Someday we can maybe add support for fixing caps.
                key = key.lower()
                data = data.lower()
                if key not in ignore_words:
                    _add_misspelling(key, data, misspellings)
                # generate alternative misspellings/fixes
                for x, table in translate_tables:
                    if x in key:
                        alt_key = key.translate(table)
                        alt_data = data.translate(table)
                        if alt_key not in ignore_words:
                            _add_misspelling(alt_key, alt_data, misspellings)


def _add_misspelling(
    key: str,
    data: str,
    misspellings: Dict[str, Misspelling],
) -> None:
    data = data.strip()

    if "," in data:
        fix = False
        data, reason = data.rsplit(",", 1)
        reason = reason.lstrip()
    else:
        fix = True
        reason = ""

    misspellings[key] = Misspelling(
        tuple(c.strip() for c in data.split(",")),
        fix,
        reason,
    )
