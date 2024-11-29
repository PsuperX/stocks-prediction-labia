from pathlib import Path
from typing import Mapping, Sequence
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV
from stock_forecasting.features import WaveletTransformer
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


import typer
from loguru import logger
from tqdm import tqdm

from stock_forecasting.config import MODELS_DIR, PROCESSED_DATA_DIR
from stock_forecasting.features import WaveletTransformer

app = typer.Typer()


class WindowGenerator:
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        label_columns: list = None,
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        inputs = features[self.input_slice, :]
        labels = features[self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, self.column_indices[name]] for name in self.label_columns], axis=-1
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([self.input_width, None])
        labels.set_shape([self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col="T (degC)", max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col[0]} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[plot_col[0]][n, :],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[label_col_index][n, :],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=None,
        )

        ds = ds.map(self.split_window)

        labels_as_dict = lambda features, labels: (
            features,
            {name[0].replace("^", "."): labels[:, i] for i, name in enumerate(self.label_columns)},
        )
        ds = ds.map(labels_as_dict)

        ds = ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def compile_and_fit(model: Model, window: WindowGenerator, epochs: int = 20, patience: int = 2):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss={name[0].replace("^", "."): "mse" for name in window.label_columns},
        optimizer=tf.keras.optimizers.Adam(),
        metrics={name[0].replace("^", "."): ["mae"] for name in window.label_columns},
    )
    # model.summary()

    history = model.fit(
        window.train, epochs=epochs, validation_data=window.val, callbacks=[early_stopping]
    )
    return history


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    input_width: int = 10,  # How many days of context
    out_steps: int = 5,  # How many days to predict
    epochs: int = 20,
):
    multi_window = WindowGenerator(
        input_width=input_width,
        label_width=out_steps,
        shift=out_steps,
        train_df=train_df,
        val_df=val_df,
        label_columns=[(ticker, "Target") for ticker in train_df.columns.levels[0]],
    )

    inputs = layers.Input(shape=(input_width, len(train_df.columns)))

    x = layers.LSTM(100, activation="relu", return_sequences=True)(inputs)
    x = layers.LSTM(100, activation="relu")(x)
    shared = layers.Dense(100, activation="relu")(x)

    outputs = []
    for name in train_df.columns.levels[0]:
        x = layers.Dense(100, activation="relu")(shared)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(out_steps, name=name.replace("^", "."))(x)
        outputs.append(x)

    model = Model(inputs=inputs, outputs=outputs)

    history = compile_and_fit(model, multi_window, epochs=epochs, patience=10)
    return history, multi_window


if __name__ == "__main__":
    app()
