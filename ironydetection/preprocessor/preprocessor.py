from typing import Iterable

import emoji
import pandas as pd

from ironydetection.preprocessor.emoticon import get_emoticon_regex, emoticon_to_emoji
from ironydetection.preprocessor.regex_expressions import *
from ironydetection.preprocessor.slang import replace_slang
from ironydetection.preprocessor.tokenizer import Tokenizer

COLUMN_MAPPER = {
    'Tweet index': 'id',
    'Label': 'label',
    'Tweet text': 'tweet'
}

# by default emoji package uses ":" as delimiter
_EMOJI_DELIMITER = ' '


__all__ = [
    'load_csv', 'DataframePreprocessor'
]


def load_csv(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path, sep='\t').rename(columns=COLUMN_MAPPER)


class DataframePreprocessor:
    def __init__(self, hashtags='keep', mention_placeholder='#mention#', link_placeholder='#link#',
                 translate_slang=True, remove_numbers=True, lemmas=True, emojis='convert_to_text'):
        self._params = {
            'hashtags': hashtags,
            'translate_slang': translate_slang,
            'mention_placeholder': mention_placeholder,
            'link_placeholder': link_placeholder,
            'remove_numbers': remove_numbers,
            'lemmas': lemmas,
            'emojis': emojis
        }
        self.validate_params()

        self._tokenize = Tokenizer()

    def validate_params(self):
        if self._params['hashtags'] not in ('keep', 'remove_#', 'remove'):
            raise ValueError('invalid value for "hashtags"')

        if self._params['emojis'] not in ('convert_to_text', 'remove'):
            raise ValueError('invalid value for "emojis"')

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.preprocess(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # copy dataframe and reindex
        df = df.copy(deep=True).reset_index(drop=True)

        # lower case all texts
        df['tweet'] = df['tweet'].str.lower()

        # add hashtags column
        df['hashtags'] = pd.Series(extract_hashtags(df['tweet'].values))

        # some hashtags are #written#like#this
        def repl(match):
            return "{} ".format(match.group())

        df['tweet'] = df['tweet'].map(lambda tweet: hashtag_re.sub(repl, tweet))

        df['emojis'] = pd.Series(extract_emojis(df['tweet'].values))

        if self._params['hashtags'] == 'remove_#':
            df['tweet'] = df['tweet'].str.replace('#', '')
        elif self._params['hashtags'] == 'remove':
            df['tweet'] = df['tweet'].map(lambda tweet: hashtag_re.sub('', tweet))

        tweets = df['tweet']

        if self._params['emojis'] == 'convert_to_text':
            # convert both the emoticons and emojis to text
            tweets = tweets.map(
                lambda tweet: emoji.demojize(emoticon_to_emoji(tweet), delimiters=(_EMOJI_DELIMITER, _EMOJI_DELIMITER))
            )
        elif self._params['emojis'] == 'remove':
            tweets = tweets.map(lambda tweet: emoji.get_emoji_regexp().sub(' ', emoticon_to_emoji(tweet)))

        if self._params['translate_slang']:
            _tweets, _n_slang = do_translate_slang(tweets.values)

            tweets = pd.Series(_tweets)

            # add column to count the slang in original text
            df['n_slang'] = pd.Series(_n_slang)

        # @todo we can extract the URLs and do something with them like fetching the page and getting the page title
        tweets = tweets.map(lambda tweet: uri_re.sub(self._params['link_placeholder'], tweet))
        tweets = tweets.map(lambda tweet: mentions_re.sub(self._params['mention_placeholder'], tweet))

        if self._params['remove_numbers']:
            tweets = tweets.map(lambda tweet: numbers_re.sub('', tweet))

        # replace | ( ) [ ] / etc. with space
        tweets = tweets.map(lambda tweet: punctuation_re.sub(' ', tweet))

        # tokenize text
        df['tweet'] = tweets.map(lambda tweet: self._tokenize(tweet, lemmas=self._params['lemmas']))

        # https://stackoverflow.com/a/31567315
        # handle NaN values
        for key in ('hashtags', 'emojis', 'tweet'):
            df.loc[df[key].isnull(), [key]] = df.loc[df[key].isnull(), key].apply(lambda x: [])

        return df

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: dict):
        for param, value in params.items():
            if param in self._params:
                self._params[param] = value

        self.validate_params()


def extract_hashtags(tweets: Iterable[str]):
    for tweet in tweets:
        yield hashtag_re.findall(tweet)


def extract_emojis(tweets: Iterable[str]):
    for tweet in tweets:
        emojis, emoticons = emoji.get_emoji_regexp().findall(tweet), get_emoticon_regex().findall(tweet)

        yield emojis + emoticons


def do_translate_slang(tweets):
    n_slang = []

    for index in range(0, len(tweets)):
        translated, count = replace_slang(tweets[index])

        tweets[index] = translated
        n_slang.append(count)

    return tweets, n_slang


if __name__ == '__main__':
    import os

    path = os.path.join(os.path.dirname(__file__), '..', '..', 'input', 'datasets', 'train', 'SemEval2018-T3-train'
                                                                                             '-taskA_emoji_irony'
                                                                                             'Hashtags.txt')

    _df = load_csv(path)

    p = DataframePreprocessor()
    print(p(_df.iloc[[0]])['tweet'])
    p = DataframePreprocessor()
    print(p(_df.iloc[[1]])['tweet'])
