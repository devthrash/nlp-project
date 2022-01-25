import re

"""
Emoticon to emoji list from:
https://stackoverflow.com/questions/29488143/is-there-an-equivalence-table-to-convert-ascii-smileys-to-unicode-emojis/29581503#29581503
"""

__all__ = ['emoticon_to_emoji', 'get_emoticon_regex']

_EMOTICON_DICT = {
    'o/': '👋',
    '</3': '💔',
    '<3': '💗',
    '8-D': '😁',
    '8D': '😁',
    ':-D': '😁',
    '=-3': '😁',
    '=-D': '😁',
    '=3': '😁',
    '=D': '😁',
    'B^D': '😁',
    'X-D': '😁',
    'XD': '😁',
    'x-D': '😁',
    'xD': '😁',
    ':\')': '😂',
    ':\'-)': '😂',
    ':-))': '😃',
    '8)': '😄',
    ':)': '😄',
    ':-)': '😄',
    ':3': '😄',
    ':D': '😄',
    ':]': '😄',
    ':^)': '😄',
    ':c)': '😄',
    ':o)': '😄',
    ':}': '😄',
    ':っ)': '😄',
    '=)': '😄',
    '=]': '😄',
    '0:)': '😇',
    '0:-)': '😇',
    '0:-3': '😇',
    '0:3': '😇',
    '0;^)': '😇',
    'O:-)': '😇',
    '3:)': '😈',
    '3:-)': '😈',
    '}:)': '😈',
    '}:-)': '😈',
    '*)': '😉',
    '*-)': '😉',
    ':-,': '😉',
    ';)': '😉',
    ';-)': '😉',
    ';-]': '😉',
    ';D': '😉',
    ';]': '😉',
    ';^)': '😉',
    ':-|': '😐',
    ':|': '😐',
    ':(': '😒',
    ':-(': '😒',
    ':-<': '😒',
    ':-[': '😒',
    ':-c': '😒',
    ':<': '😒',
    ':[': '😒',
    ':c': '😒',
    ':{': '😒',
    ':っC': '😒',
    '%)': '😖',
    '%-)': '😖',
    ':-P': '😜',
    ':-b': '😜',
    ':-p': '😜',
    ':-Þ': '😜',
    ':-þ': '😜',
    ':P': '😜',
    ':b': '😜',
    ':p': '😜',
    ':Þ': '😜',
    ':þ': '😜',
    ';(': '😜',
    '=p': '😜',
    'X-P': '😜',
    'XP': '😜',
    'd:': '😜',
    'x-p': '😜',
    'xp': '😜',
    ':-||': '😠',
    ':@': '😠',
    ':-.': '😡',
    ':-/': '😡',
    ':/': '😡',
    ':L': '😡',
    ':S': '😡',
    ':\\': '😡',
    '=/': '😡',
    '=L': '😡',
    '=\\': '😡',
    ':\'(': '😢',
    ':\'-(': '😢',
    '^5': '😤',
    '^<_<': '😤',
    'o/\\o': '😤',
    '|-O': '😫',
    '|;-)': '😫',
    ':###..': '😰',
    ':-###..': '😰',
    'D-\':': '😱',
    'D8': '😱',
    'D:': '😱',
    'D:<': '😱',
    'D;': '😱',
    'D=': '😱',
    'DX': '😱',
    'v.v': '😱',
    '8-0': '😲',
    ':-O': '😲',
    ':-o': '😲',
    ':O': '😲',
    ':o': '😲',
    'O-O': '😲',
    'O_O': '😲',
    'O_o': '😲',
    'o-o': '😲',
    'o_O': '😲',
    'o_o': '😲',
    ':$': '😳',
    '#-)': '😵',
    ':#': '😶',
    ':&': '😶',
    ':-#': '😶',
    ':-&': '😶',
    ':-X': '😶',
    ':X': '😶',
    ':-J': '😼',
    ':*': '😽',
    ':^*': '😽',
    'ಠ_ಠ': '🙅',
    '*\\0/*': '🙆',
    '\\o/': '🙆',
    ':>': '😄',
    '>.<': '😡',
    '>:(': '😠',
    '>:)': '😈',
    '>:-)': '😈',
    '>:/': '😡',
    '>:O': '😲',
    '>:P': '😜',
    '>:[': '😒',
    '>:\\': '😡',
    '>;)': '😈',
    '>_>^': '😤'
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
