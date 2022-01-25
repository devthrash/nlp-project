import re
from os import path


_SLANG_DICT = None
_COMPILED_REGEX = []

__all__ = [
    'get_slang_dict', 'replace_slang'
]


def get_slang_dict():
    global _SLANG_DICT

    if _SLANG_DICT is None:
        slang_file_path = path.join(path.dirname(__file__), '..', '..', 'input', 'slang', 'slang.txt')

        slang_dict = {}
        with open(slang_file_path) as slang:
            for line in slang:
                slang, translation = line.split('=')

                slang_dict[slang.lower()] = translation.lower().replace('\n', '')

        _SLANG_DICT = slang_dict

    return _SLANG_DICT


def _compile_regex():
    # compile all regexes when called for the first time
    global _COMPILED_REGEX

    if not len(_COMPILED_REGEX):
        for slang in get_slang_dict().keys():
            _COMPILED_REGEX.append(re.compile(r'(?<!\w)' + re.escape(slang) + r'(?!\w)'))


def replace_slang(text: str):
    _compile_regex()

    count = 0
    for regex, translation in zip(_COMPILED_REGEX, get_slang_dict().values()):
        text, n_subs = regex.subn(translation, text)

        count += n_subs

    return text, count
