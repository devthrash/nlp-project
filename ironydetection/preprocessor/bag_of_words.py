from tqdm import tqdm

__all__ = ['BagOfWords']


class BagOfWords:
    """
    I implemented BaW myself because CountVectorizer from sklearn creates sparse matrix that are very slow and use a
    lot of memory when converted to dense matrix
    """
    def __init__(self, corpus, ngram_range=(1, 1), show_progress=False):
        self._ngram_range = ngram_range
        self._vocabulary = {}

        for ngram in range(ngram_range[0], ngram_range[1] + 1):
            self._build_ngram_vocabulary(corpus, ngram, show_progress)

    @property
    def ngram_range(self):
        return self._ngram_range

    def _build_ngram_vocabulary(self, corpus, n, show_progress):
        if n < 1:
            raise ValueError('cannot build ngrams with n < 1')

        for tokenized_sentence in tqdm(corpus, disable=not show_progress):
            for i in range(0, len(tokenized_sentence) + 1 - n):
                word = []
                for j in range(i, i + n):
                    word.append(tokenized_sentence[j])

                self._increment_vocabulary(' '.join(word))

    def _increment_vocabulary(self, word):
        if word in self._vocabulary:  # existing word, increment count
            self._vocabulary[word] += 1
        else:
            self._vocabulary[word] = 1  # new word, add to dictionary

    def get_vocabulary(self):
        return self._vocabulary

    def count(self, word):
        if word in self._vocabulary:
            return self._vocabulary[word]
        else:
            return 0


if __name__ == '__main__':
    text = [
        'This is a sentence',
        'This is another sentence',
        'This is the third sentence and the last one'
    ]

    bow = BagOfWords(text)
    print('count for "this"={}'.format(bow.count('this')))
    print('count for "sentence"={}'.format(bow.count('sentence')))
    print('count for "another"={}'.format(bow.count('another')))
    print(bow.get_vocabulary())

    bow = BagOfWords(text, ngram_range=(1, 2))
    print('count for "this is"={}'.format(bow.count('this is')))
    print(bow.get_vocabulary())
