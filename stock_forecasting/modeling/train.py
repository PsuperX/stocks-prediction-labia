from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import BaseCrossValidator
import numpy as np


import typer
from loguru import logger
from tqdm import tqdm

from stock_forecasting.config import MODELS_DIR, PROCESSED_DATA_DIR
from stock_forecasting.features import WaveletTransformer

app = typer.Typer()


class CustomTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, train_size=None):
        self.n_splits = n_splits
        self.train_size = train_size
    
    def split(self, X, y=None, groups=None):
        # Check that the input data is at least of the required size
        n_samples = len(X)
        
        if self.train_size is None:
            self.train_size = n_samples // (self.n_splits + 1)
        
        # Ensure we have enough samples to perform the split
        if n_samples <= self.train_size * self.n_splits:
            raise ValueError("The number of samples is too small for the number of splits.")
        
        # Generate the splits
        for i in range(self.n_splits):
            train_end = self.train_size * (i + 1)  # The end of the training set
            test_start = train_end  # The start of the test set
            
            if test_start + self.train_size > n_samples:
                test_start = n_samples - self.train_size

            train_indices = np.arange(train_end)
            test_indices = np.arange(test_start, min(test_start + self.train_size, n_samples))
            
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits




@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    X, y = make_regression(n_samples=100, n_features=100, n_targets=1)

    pipeline = Pipeline([
        ("preprocessing", StandardScaler()),
        ("wavelet", WaveletTransformer())
        #("classifier", )
    ])

    pipeline.fit(X[:-10], y[:-10])





if __name__ == "__main__":
    app()