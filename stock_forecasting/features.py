from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import typer
from loguru import logger
from tqdm import tqdm

from .config import PROCESSED_DATA_DIR
from .utils import wavelet_transform, waveletSmooth

app = typer.Typer()

class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, level=1, wavelet='db1'):
        self.level = level
        self.wavelet = wavelet

        self.prev_x = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.prev_x is None:
            X_transformed = np.copy(X)
            for i in range(X.shape[1]):
                X_transformed[:, i] = waveletSmooth(X[:, i], level=self.level, wavelet=self.wavelet)[-len(X):]

            self.prev_x = X_transformed
            return X_transformed
        
        else:
            # for the validation we have to do the transform using training data + the current and past validation data
            # i.e. we CAN'T USE all the validation data because we would then look into the future 
            temp = np.copy(self.prev_x)
            X_test_WT = np.copy(X)
            for j in range(X.shape[0]):
                #first concatenate train with the latest validation sample
                temp = np.append(temp, np.expand_dims(X[j,:], axis=0), axis=0)
                for i in range(X.shape[1]):
                    X_test_WT[j,i] = waveletSmooth(temp[:,i], level=1)[-1]

            return X_test_WT


if __name__ == "__main__":
    app()