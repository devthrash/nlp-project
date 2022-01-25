import re
import string

__all__ = [
    'hashtag_re', 'uri_re', 'mentions_re', 'trim_re', 'punctuation_re', 'numbers_re'
]

hashtag_re = re.compile(r'(#[a-z0-9_]+)')

uri_re = re.compile(r'(https?://'                   # protocol
                    r'[-a-zA-Z0-9@:%._+~#=]*'       # domain
                    r'[-a-zA-Z0-9@:%_+.~#?&/=]*)')  # path, query parameters etc

mentions_re = re.compile(r'(?:@[a-z0-9_]{1,15})')  # twitter mentions

punctuation_re = re.compile(r'[' + re.escape(string.punctuation.replace('#', '').replace('\'', '')) + r']+')

numbers_re = re.compile(r'[0-9]+')

trim_re = re.compile(r'\s+')
