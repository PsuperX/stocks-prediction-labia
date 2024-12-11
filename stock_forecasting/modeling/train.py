from pathlib import Path
from typing import Mapping, Sequence, Optional, Tuple
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
from tensorflow.keras import regularizers
from dataclasses import dataclass
import pickle
import os


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
    def split_window(self, features, filter_features: bool = True):
        inputs = features[self.input_slice, :]
        labels = features[self.labels_slice, :]
        if self.feature_columns is not None and filter_features:
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
            ax.axhline(y=0, color="red", linestyle=":", linewidth=1)

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
                inputs_filtered = tf.stack(
                    [inputs[:, :, self.column_indices[name]] for name in self.feature_columns],
                    axis=-1,
                )
                predictions = model(inputs_filtered)
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

    def make_dataset(self, data, sequence_stride: int = 1, filter_features: bool = True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=sequence_stride,
            shuffle=False,
            batch_size=None,
        )

        ds = ds.map(lambda x: self.split_window(x, filter_features))

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
            result = next(iter(self.make_dataset(self.train_df, filter_features=False)))
            # And cache it for next time
            self._example = result
        return result


@dataclass
class LSTMManager:
    window: WindowGenerator
    settings: dict
    test_number: int

    def fit_model(
        self,
        model: Model,
        epochs: int = 20,
        patience: int = 10,
        verbose: int = "auto",
    ):
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min", restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=patience
        )
        checkpoint_filepath = MODELS_DIR / f"test{self.test_number}" / "best_so_far.keras"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            mode="max",
            save_best_only=True,
        )

        # Fit
        train = self.window.train
        val = self.window.val
        history = model.fit(
            train,
            epochs=epochs,
            validation_data=val,
            callbacks=[early_stopping, reduce_lr, model_checkpoint_callback],
            verbose=verbose,
        )
        os.makedirs(MODELS_DIR / f"test{self.test_number}", exist_ok=True)
        with open(MODELS_DIR / f"test{self.test_number}" / "history.pkl", "wb") as f:
            pickle.dump(history, f)
        return history

    def train_lstm(
        self,
        epochs: int = 20,
        verbose: int = "auto",
    ):
        # Create shared network
        inputs = layers.Input(shape=(self.window.input_width, len(self.window.feature_columns)))

        x = layers.LSTM(self.settings["lstm_sizes"][0], return_sequences=True)(inputs)
        for size in self.settings["lstm_sizes"][1:-1]:
            x = layers.LSTM(size, return_sequences=True)(x)
        x = layers.LSTM(self.settings["lstm_sizes"][-1])(x)
        x = layers.Dropout(self.settings["dropout"])(x)
        shared = layers.Dense(
            self.settings["shared_dense"],
            activation="relu",
            kernel_regularizer=regularizers.l2(self.settings["l2"]),
        )(x)

        # Create a head for each stock
        outputs = []
        for name in self.window.train_df.columns.levels[0]:
            x = layers.Dense(
                self.settings["per_stock_sizes"][0],
                activation="relu",
                kernel_regularizer=regularizers.l2(self.settings["l2"]),
            )(shared)
            x = layers.Dropout(self.settings["dropout"])(x)
            for size in self.settings["per_stock_sizes"][1:]:
                x = layers.Dense(
                    size,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(self.settings["l2"]),
                )(x)
                x = layers.Dropout(self.settings["dropout"])(x)
            x = layers.Dense(self.window.label_width, name=name.replace("^", "."))(x)
            outputs.append(x)

        # Combine all networks
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss={name[0].replace("^", "."): "mse" for name in self.window.label_columns},
            optimizer=tf.keras.optimizers.Adam(self.settings["lr"]),
            metrics={
                name[0].replace("^", "."): ["mse", "R2Score"] for name in self.window.label_columns
            },
        )
        # model.summary()

        # Train the model
        history = self.fit_model(model, epochs=epochs, patience=10, verbose=verbose)
        return history

    def plot_predictions(self, model: Model = None, plot_col=("COST", "Target")):
        if model is None:
            self.window.plot(model=self.best_model, plot_col=plot_col)
        else:
            self.window.plot(model=model, plot_col=plot_col)

    def predict(
        self,
        model: Model,
        true_prices: pd.DataFrame,
        scaler: RobustScaler,
    ) -> pd.DataFrame:
        # Predict on test data
        # predictions are percentage change
        pred = model.predict(self.window.test)

        # Reshape predictions
        pred = np.array(pred)
        pred = pred.reshape(-1, pred.shape[0])

        # Undo scaling
        pred = scaler.inverse_transform(pred)

        pred = pd.DataFrame(
            pred,
            columns=[ticker for ticker, _ in self.window.label_columns],
            index=self.window.test_df.index[self.window.input_width :],
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

    @property
    def best_model(self) -> Optional[Model]:
        path = MODELS_DIR / f"test{self.test_number}" / "best_so_far.keras"
        if path.exists():
            return tf.keras.models.load_model(path)
        return None


def prepare_data(
    df: pd.DataFrame,
    validation_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    # Train-Test split
    train_df = df[df.index < "2024-01-01"]
    test_df = df[df.index >= "2023-12-15"]

    # Train-Val split
    train_size = int(len(train_df) * (1 - validation_size))
    val_df = train_df.iloc[train_size:]
    train_df = train_df.iloc[:train_size]

    # Normalize data
    scaler = RobustScaler()
    scaler.set_output(transform="pandas")

    # Fit on train data
    columns = train_df.columns
    train_df = scaler.fit_transform(train_df)
    train_df.columns = columns

    # Transform validation data
    columns = val_df.columns
    val_df = scaler.transform(val_df)
    val_df.columns = columns

    # Transform test data
    label_scaler = RobustScaler()
    label_scaler.fit(test_df.xs("Target", level="Price", axis=1))

    columns = test_df.columns
    test_df = scaler.transform(test_df)
    test_df.columns = columns

    return train_df, val_df, test_df, label_scaler


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
