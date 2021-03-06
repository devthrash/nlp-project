import re

"""
Emoticon to emoji list from:
https://stackoverflow.com/questions/29488143/is-there-an-equivalence-table-to-convert-ascii-smileys-to-unicode-emojis/29581503#29581503
"""

__all__ = ['emoticon_to_emoji', 'get_emoticon_regex']

_EMOTICON_DICT = {
    'o/': '๐',
    '</3': '๐',
    '<3': '๐',
    '8-D': '๐',
    '8D': '๐',
    ':-D': '๐',
    '=-3': '๐',
    '=-D': '๐',
    '=3': '๐',
    '=D': '๐',
    'B^D': '๐',
    'X-D': '๐',
    'XD': '๐',
    'x-D': '๐',
    'xD': '๐',
    ':\')': '๐',
    ':\'-)': '๐',
    ':-))': '๐',
    '8)': '๐',
    ':)': '๐',
    ':-)': '๐',
    ':3': '๐',
    ':D': '๐',
    ':]': '๐',
    ':^)': '๐',
    ':c)': '๐',
    ':o)': '๐',
    ':}': '๐',
    ':ใฃ)': '๐',
    '=)': '๐',
    '=]': '๐',
    '0:)': '๐',
    '0:-)': '๐',
    '0:-3': '๐',
    '0:3': '๐',
    '0;^)': '๐',
    'O:-)': '๐',
    '3:)': '๐',
    '3:-)': '๐',
    '}:)': '๐',
    '}:-)': '๐',
    '*)': '๐',
    '*-)': '๐',
    ':-,': '๐',
    ';)': '๐',
    ';-)': '๐',
    ';-]': '๐',
    ';D': '๐',
    ';]': '๐',
    ';^)': '๐',
    ':-|': '๐',
    ':|': '๐',
    ':(': '๐',
    ':-(': '๐',
    ':-<': '๐',
    ':-[': '๐',
    ':-c': '๐',
    ':<': '๐',
    ':[': '๐',
    ':c': '๐',
    ':{': '๐',
    ':ใฃC': '๐',
    '%)': '๐',
    '%-)': '๐',
    ':-P': '๐',
    ':-b': '๐',
    ':-p': '๐',
    ':-ร': '๐',
    ':-รพ': '๐',
    ':P': '๐',
    ':b': '๐',
    ':p': '๐',
    ':ร': '๐',
    ':รพ': '๐',
    ';(': '๐',
    '=p': '๐',
    'X-P': '๐',
    'XP': '๐',
    'd:': '๐',
    'x-p': '๐',
    'xp': '๐',
    ':-||': '๐ ',
    ':@': '๐ ',
    ':-.': '๐ก',
    ':-/': '๐ก',
    ':/': '๐ก',
    ':L': '๐ก',
    ':S': '๐ก',
    ':\\': '๐ก',
    '=/': '๐ก',
    '=L': '๐ก',
    '=\\': '๐ก',
    ':\'(': '๐ข',
    ':\'-(': '๐ข',
    '^5': '๐ค',
    '^<_<': '๐ค',
    'o/\\o': '๐ค',
    '|-O': '๐ซ',
    '|;-)': '๐ซ',
    ':###..': '๐ฐ',
    ':-###..': '๐ฐ',
    'D-\':': '๐ฑ',
    'D8': '๐ฑ',
    'D:': '๐ฑ',
    'D:<': '๐ฑ',
    'D;': '๐ฑ',
    'D=': '๐ฑ',
    'DX': '๐ฑ',
    'v.v': '๐ฑ',
    '8-0': '๐ฒ',
    ':-O': '๐ฒ',
    ':-o': '๐ฒ',
    ':O': '๐ฒ',
    ':o': '๐ฒ',
    'O-O': '๐ฒ',
    'O_O': '๐ฒ',
    'O_o': '๐ฒ',
    'o-o': '๐ฒ',
    'o_O': '๐ฒ',
    'o_o': '๐ฒ',
    ':$': '๐ณ',
    '#-)': '๐ต',
    ':#': '๐ถ',
    ':&': '๐ถ',
    ':-#': '๐ถ',
    ':-&': '๐ถ',
    ':-X': '๐ถ',
    ':X': '๐ถ',
    ':-J': '๐ผ',
    ':*': '๐ฝ',
    ':^*': '๐ฝ',
    'เฒ _เฒ ': '๐',
    '*\\0/*': '๐',
    '\\o/': '๐',
    ':>': '๐',
    '>.<': '๐ก',
    '>:(': '๐ ',
    '>:)': '๐',
    '>:-)': '๐',
    '>:/': '๐ก',
    '>:O': '๐ฒ',
    '>:P': '๐',
    '>:[': '๐',
    '>:\\': '๐ก',
    '>;)': '๐',
    '>_>^': '๐ค'
}

_COMPILED_REGEX = []
_COMPILED_MATCH_ALL_REGEX = None


def _compile_regex():
    # compile all regexes when called for the first time
    global _COMPILED_REGEX

    if not len(_COMPILED_REGEX):
        for emoticon in _EMOTICON_DICT.keys():
            _COMPILED_REGEX.append(re.compile(r'(?<!\w)' + re.escape(emoticon) + r'(?!\w)'))


def emoticon_to_emoji(string: str):
    _compile_regex()

    # substitute emoticons for emojis
    for regex, emoji in zip(_COMPILED_REGEX, _EMOTICON_DICT.values()):
        string = regex.sub(emoji, string)

    return string


def get_emoticon_regex():
    global _COMPILED_MATCH_ALL_REGEX

    if _COMPILED_MATCH_ALL_REGEX is None:
        _compile_regex()
        _COMPILED_MATCH_ALL_REGEX = re.compile(r'(' + r'|'.join([expr.pattern for expr in _COMPILED_REGEX]) + r')')

    return _COMPILED_MATCH_ALL_REGEX


if __name__ == '__main__':
    print(emoticon_to_emoji('Haha good day :)no? :('))
    print(u'\U0001F624')
    print(get_emoticon_regex().findall('this is an emoticon :-D'))
    print(get_emoticon_regex().findall('Sweet United Nations video. Just in time for Christmas. #imagine #NoReligion #irony http://t.co/fej2v3OUBR'.lower()))
