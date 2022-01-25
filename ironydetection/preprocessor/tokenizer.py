from nltk import download
from nltk.stem import WordNetLemmatizer

__all__ = ['Tokenizer']


class Tokenizer:
    def __init__(self):
        download_files()

        self._lemmatizer = WordNetLemmatizer()

    def __call__(self, text: str, **kwargs):
        return self._tokenize(text, **kwargs)

    def _tokenize(self, text: str, lemmas=True) -> list:
        tokens = [tk for tk in text.split(' ') if tk != '']
        if lemmas:
            tokens = [self._lemmatizer.lemmatize(tk) for tk in tokens]

        return tokens


def download_files():
    download('wordnet', quiet=True)
    download('omw-1.4', quiet=True)
