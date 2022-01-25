import numpy as np
from torch.utils.data import Dataset

from ironydetection.net.encoder import BaseEncoder
from ironydetection.preprocessor import load_csv, DataframePreprocessor

__all__ = [
    'TweetsDataset'
]


class TweetsDataset(Dataset):
    """
    Reads CSV files
    Preprocesses and tokenizes text
    Convert tokens to word vector's indices
    """
    def __init__(self, dataset: str, encoder: BaseEncoder,
                 preprocessor: DataframePreprocessor = DataframePreprocessor(), df=None):

        if df is not None:
            self._df = df
        else:
            self._df = load_csv(dataset)
        self._preprocessor = preprocessor
        self._encoder = encoder

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        # read one entry from the dataframe
        df = self._preprocessor.preprocess(self._df.iloc[[index]])

        tweet = df['tweet'].values[0]

        return np.array([self._encoder.encode(tk) for tk in tweet]), len(tweet), df['label'].values[0]
