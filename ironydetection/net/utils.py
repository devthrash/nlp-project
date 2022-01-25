from typing import Tuple

import torch
from gensim.models import KeyedVectors
import gensim.downloader
import numpy as np
from torch.utils.data import DataLoader

from ironydetection.net.encoder import GensimWordVectorEncoder

__all__ = ['get_pretrained_embeddings_and_create_encoder', 'dataloader']


def get_pretrained_embeddings_and_create_encoder(name='glove-twitter-25') -> Tuple[torch.FloatTensor,
                                                                                   GensimWordVectorEncoder]:
    wv: KeyedVectors = gensim.downloader.load(name)
    word_vectors = np.append(wv.vectors, _generate_unknown_vector(wv.vectors.shape), axis=0)

    return torch.FloatTensor(word_vectors), GensimWordVectorEncoder(wv, unknown_index=len(word_vectors) - 1)


def _generate_unknown_vector(shape):
    np.random.seed(0)

    return np.random.rand(1, shape[1])


def _pad_batch_data(data):
    # @todo sort vectors by length
    indices, lengths, labels = zip(*data)

    max_len = max(lengths)
    indices_padded = torch.zeros((len(data), max_len))

    for i in range(len(data)):
        for j in range(max_len):
            if j < len(indices[i]):
                indices_padded[i][j] = indices[i][j]
            else:
                break

    return indices_padded.long(), torch.tensor(lengths).long(), torch.tensor(labels).float()


def dataloader(dataset, shuffle=True, batch_size=64) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_pad_batch_data, num_workers=0)
