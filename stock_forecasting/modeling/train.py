from pathlib import Path
from typing import Mapping, Sequence
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV
from stock_forecasting.features import WaveletTransformer
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
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


# Adapted from https://www.tensorflow.org/tutorials/structured_data/time_series
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
        feature_columns: list = None,
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.feature_columns = feature_columns
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

    @tf.function
    def split_window(self, features):
        inputs = features[self.input_slice, :]
        labels = features[self.labels_slice, :]
        if self.feature_columns is not None:
            inputs = tf.stack(
                [inputs[:, self.column_indices[name]] for name in self.feature_columns], axis=-1
            )
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, self.column_indices[name]] for name in self.label_columns], axis=-1
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([self.input_width, None])
        labels.set_shape([self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col=("A", "Target"), max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            ax = plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col[0]} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )
            ax.axhline(y=0, color="red", linestyle=":", linewidth=2)

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

    def make_dataset(self, data, sequence_stride: int = 1):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=sequence_stride,
            shuffle=False,
            batch_size=None,
        )

        ds = ds.map(self.split_window)

        # Turn labels to dictionaries for multi-task training
        labels_as_dict = lambda features, labels: (
            features,
            {name[0].replace("^", "."): labels[:, i] for i, name in enumerate(self.label_columns)},
        )
        ds = ds.map(labels_as_dict)

        # This speeds-up stuff
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
        return self.make_dataset(self.test_df, sequence_stride=self.label_width)

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


def compile_and_fit(
    model: Model,
    window: WindowGenerator,
    epochs: int = 20,
    patience: int = 10,
    verbose: int = "auto",
):
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=patience
    )
    checkpoint_filepath = MODELS_DIR / "best_so_far.keras"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_loss",
        mode="max",
        save_best_only=True,
    )

    model.compile(
        loss={name[0].replace("^", "."): "mse" for name in window.label_columns},
        optimizer=tf.keras.optimizers.Adam(),
        metrics={name[0].replace("^", "."): ["mae"] for name in window.label_columns},
    )
    # model.summary()

    history = model.fit(
        window.train,
        epochs=epochs,
        validation_data=window.val,
        callbacks=[early_stopping, reduce_lr, model_checkpoint_callback],
        verbose=verbose,
    )
    return history


def train_lstm(
    multi_window: WindowGenerator,
    epochs: int = 20,
    verbose: int = "auto",
):
    # Create shared network
    inputs = layers.Input(shape=(multi_window.input_width, len(multi_window.train_df.columns)))

    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.5)(x)
    shared = layers.Dense(100, activation="relu")(x)

    # Create a head for each stock
    outputs = []
    for name in multi_window.train_df.columns.levels[0]:
        x = layers.Dense(50, activation="relu")(shared)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(multi_window.label_width, name=name.replace("^", "."))(x)
        outputs.append(x)

    # Combine all networks
    model = Model(inputs=inputs, outputs=outputs)

    # Train the model
    history = compile_and_fit(model, multi_window, epochs=epochs, patience=10, verbose=verbose)
    return history


def predict(
    model: Model, window: WindowGenerator, true_prices: pd.DataFrame, scaler: RobustScaler
) -> pd.DataFrame:
    # Predict on test data
    # predictions are percentage change
    pred = model.predict(window.test)

    # Reshape predictions
    pred = np.array(pred)
    pred = pred.reshape(-1, pred.shape[0])

    # Undo scaling
    pred = scaler.inverse_transform(pred)

    pred = pd.DataFrame(
        pred,
        columns=[ticker for ticker, _ in window.label_columns],
        index=window.test_df.index[window.input_width :],
    )

    # Calculate predicted close price
    orig = true_prices.sort_index(axis=1)
    result = np.zeros_like(orig)
    for i in range(0, len(orig), 5):
        prev = orig.iloc[i]
        for j in range(5):
            if i + j >= len(result):
                break
            result[i + j] = prev * (1 + pred.iloc[i + j])
            prev = result[i + j]

    result = pd.DataFrame(result, columns=pred.columns, index=pred.index)
    return result


def evaluate_last_baseline(window: WindowGenerator):
    # Layer that always predicts the last value
    class MultiStepLastBaseline(tf.keras.Model):
        def __init__(self, label_index, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label_index = label_index

        def call(self, inputs):
            return tf.tile(inputs[:, -1:, self.label_index], [1, window.label_width])

    # Create shared inputs
    inputs = layers.Input(shape=(window.input_width, len(window.train_df.columns)))

    # Create a head for each stock
    outputs = []
    for name in window.train_df.columns.levels[0]:
        x = MultiStepLastBaseline(
            window.column_indices[(name, "Target")], name=name.replace("^", ".")
        )(inputs)
        outputs.append(x)

    # Combine all networks
    baseline = Model(inputs=inputs, outputs=outputs)
    baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics={name[0].replace("^", "."): ["mae"] for name in window.label_columns},
    )

    # Evaluate
    baseline_results = baseline.evaluate(window.test, verbose=0, return_dict=True)
    return baseline, baseline_results


if __name__ == "__main__":
    app()
