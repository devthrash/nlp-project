from gensim.models import KeyedVectors

__all__ = ['GensimWordVectorEncoder', 'BaseEncoder', 'Encoder']


class BaseEncoder:
    def __init__(self, dictionary, unknown_index):
        self._dictionary = dictionary
        self._unknown_index = unknown_index

    def encode(self, token):
        if token in self._dictionary:
            return self._dictionary[token]
        else:
            return self._unknown_index

    def vocab_size(self) -> int:
        return len(self._dictionary.keys())


class GensimWordVectorEncoder(BaseEncoder):
    def __init__(self, wv: KeyedVectors, unknown_index):
        super(GensimWordVectorEncoder, self).__init__(wv.key_to_index, unknown_index)


class Encoder(BaseEncoder):
    def __init__(self, vocabulary: list, unknown_token='unknown'):
        if unknown_token in vocabulary:
            vocabulary.append(unknown_token)

        dictionary = self._build_dict(vocabulary)

        super(Encoder, self).__init__(dictionary, dictionary[unknown_token])

    @staticmethod
    def _build_dict(vocabulary):
        dictionary = {}

        for token in set(vocabulary):
            dictionary[token] = len(dictionary.keys())

        return dictionary
