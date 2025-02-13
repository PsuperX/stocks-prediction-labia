import pywt
from statsmodels.robust import mad
import numpy as np
from typing import Tuple


def waveletSmooth(x, wavelet="haar", level=2, declevel=2):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="periodization", level=declevel, axis=0)
    # calculate a threshold
    sigma = mad(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec(coeff, wavelet, mode="periodization", axis=0)
    return y


# TODO: will this be used?
def wavelet_transform(X_train: np.array, X_test: np.array) -> Tuple[np.array, np.array]:
    for i in range(X_train.shape[1]):
        X_train[:, i] = waveletSmooth(X_train[:, i], level=1)[-len(X_train) :]

    # for the validation we have to do the transform using training data + the current and past validation data
    # i.e. we CAN'T USE all the validation data because we would then look into the future
    temp = np.copy(X_train)
    X_test_WT = np.copy(X_test)
    for j in range(X_test.shape[0]):
        # first concatenate train with the latest validation sample
        temp = np.append(temp, np.expand_dims(X_test[j, :], axis=0), axis=0)
        for i in range(X_test.shape[1]):
            X_test_WT[j, i] = waveletSmooth(temp[:, i], level=1)[-1]

    return X_train, X_test_WT


def calculate_lstm_stats(history):
    print("Val loss:", history["loss"][-1])
    print(
        "Mean mae:",
        np.mean(list(map(lambda x: x[1], filter(lambda x: "mae" in x[0], history.items())))),
    )
    print(
        "Mean R²:",
        np.mean(list(map(lambda x: x[1], filter(lambda x: "R2" in x[0], history.items())))),
    )
