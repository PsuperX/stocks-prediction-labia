import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingRegressor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re
import pickle
import os
from stock_forecasting.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR, INTERIM_DATA_DIR
import gc

models_dir = "../../models"

def time_series_cv(X, y, ticker, train_size=0.9, horizon=5):
    """
    Perform time series cross-validation and return model performance statistics.
    
    Arguments:
        X (pd.DataFrame): Feature matrix for the dataset.
        y (pd.Series): Target variable corresponding to X.
        ticker (str): Ticker symbol for which the model is trained.
        train_size (float, optional): Proportion of data used for training (default is 0.9).
        horizon (int, optional): Number of days to forecast (default is 5).
    
    Returns:
        dict: A summary of model performance statistics including:
            - mean_error: Mean squared error across all folds.
            - std_error: Standard deviation of mean squared error across all folds.
            - mean_r2: Mean R-squared value across all folds.
            - std_r2: Standard deviation of R-squared value across all folds.
            - feature_importances: Feature importances from the trained model.
    """
    n = len(y)
    train_len = int(n * train_size)
    
    results = []
    ticker_dir = os.path.join(models_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    idx = 0
    # Iterate through the dataset to create folds
    for start_idx in range(0, n - train_len - horizon + 1, horizon):
        train_end_idx = start_idx + train_len
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + horizon

        # Split X and y into training and testing sets
        train_X = X.iloc[start_idx:train_end_idx]
        train_y = y.iloc[start_idx:train_end_idx]
        test_X = X.iloc[test_start_idx:test_end_idx]
        test_y = y.iloc[test_start_idx:test_end_idx]

        # Retrieve start and end dates from the index
        train_start_date = train_X.index[0]
        train_end_date = train_X.index[-1]
        if train_end_date in X[X.index >= '2024-01-01'].index:
            break
        test_start_date = test_X.index[0]
        test_end_date = test_X.index[-1]

        print(f"  Training: {train_start_date} to {train_end_date}")
        print(f"  Testing: {test_start_date} to {test_end_date}")


        # Initialize scalers
        scaler = RobustScaler()
        
        # Fit scaler on train data and transform train and test data
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        #print(scaler.feature_names_in_)

        # Fit model on scaled data
        model = XGBRegressor()
        model.fit(train_X, train_y)

        model_path = os.path.join(ticker_dir, f"model_{idx}.pkl")
        scaler_path = os.path.join(ticker_dir, f"scaler_{idx}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            print(f"Model for {ticker}, fold {idx}, saved at {model_path}.")

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Predict and evaluate performance
        predictions = model.predict(test_X)
        error = mean_squared_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)
        print(f"  Error: {error:.4f}")

        # Save fold results
        results.append({
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "mse": error,
            "r2_score": r2
        })

        idx += 1

    # Calculate mean target value and overall error statistics
    mean_error = np.mean([r["mse"] for r in results])
    std_error = np.std([r["mse"] for r in results])
    mean_r2 = np.mean([r["r2_score"] for r in results])
    std_r2 = np.std([r["r2_score"] for r in results])

    # Feature importance from XGBoost
    feature_importances = model.feature_importances_

    # Summary results
    summary = {
        "ticker": ticker,
        "mean_error": mean_error,
        "std_error": std_error,
        "feature_importances": feature_importances,
        "mean_r2": mean_r2,
        "std_r2": std_r2
    }
        
    
    return summary

#========================================================Ensemble==================================================================

def ensemble(ticker_dir, models_dir):
    """
    Creates an ensemble model by combining previously saved models for a specific ticker.
    
    Arguments:
        ticker_dir (str): Directory containing the ticker's models.
        models_dir (str): Base directory where models are stored.
    
    Returns:
        ensemble_model (VotingRegressor): An ensemble model using the individual models.
    """
    #ensemble_models = {}

    ticker_path = os.path.join(models_dir, ticker_dir)

    if os.path.isdir(ticker_path):
        models = []

        for model_file in os.listdir(ticker_path):
            if model_file.endswith('.pkl') and not model_file.startswith('scaler_'):
                model_path = os.path.join(ticker_path, model_file)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    models.append((model_file, model))

        ensemble_model = VotingRegressor(estimators=models)
        result = ensemble_model.estimators[0][0]
        print(f"Ensemble Model for ticker {ticker_dir} created")


        del models
        del ensemble_model
        gc.collect()

    return result

#=================================================Predict===============================================================

def predict_xgb(X_test, ticker, ensemble_models):
    """
    Predict the target values using the saved model for a specific ticker.
    
    Arguments:
        X_test (pd.DataFrame): Test feature matrix.
        ticker (str): Ticker symbol for which the prediction is made.
        ensemble_models (dict): A dictionary of models indexed by ticker.
    
    Returns:
        predictions (np.ndarray): Predicted target values for the given test set.
    """
    model_file = ensemble_models[str(ticker)]
    model_path = os.path.join(f'../../models/{ticker}', model_file)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    idx = model_file.split('_')[1].split('.')[0]
    scaler_file = f"scaler_{idx}.pkl"
    scaler_path = os.path.join(f'../../models/{ticker}', scaler_file)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X_test)
    predictions = model.predict(X_scaled)
    
    return predictions

def true_labels_xgb(y_test):
    """
    Convert the true labels (y_test) into a list format.
    
    Arguments:
        y_test (pd.Series): True target values for the test set.
    
    Returns:
        list: A list of true target values from y_test.
    """
    res = []
    for label in y_test:
        res.append(label)
    return res


def get_test_data_for_ticker(df, ticker, target_column="Target"):
    """
    Extract the test dataset for a specific ticker.
    
    Arguments:
        df (pd.DataFrame): Full dataset containing all tickers.
        ticker (str): The ticker symbol to extract data for.
        target_column (str, optional): The name of the target column (default is 'Target').
    
    Returns:
        X_test (pd.DataFrame): Feature matrix for the test set of the given ticker.
        y_test (pd.Series): Target variable for the test set of the given ticker.
    """
    # Select data for the specific ticker
    df_ticker = df.xs(ticker, level="Ticker", axis=1)
    #print(df_ticker)

    # Filter by date range
    test_df = df_ticker[df_ticker.index >= '2024-01-01']
    #print(len(test_df))
    #print(test_df)

    # Separate features and target
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column].shift(-1).dropna()

    return X_test, y_test


def predicted_close_price(true_prices, predictions, filtered_df):
    """
    Transform predicted percentage changes into predicted closing prices.
    
    Arguments:
        true_prices (pd.DataFrame): True closing prices to use for transformation.
        predictions (np.ndarray): Predicted percentage changes for the closing prices.
        filtered_df (pd.DataFrame): Filtered DataFrame to retrieve relevant date indices.
    
    Returns:
        result (pd.DataFrame): Predicted closing prices, indexed by date.
    """
    orig = true_prices.sort_index(axis=1)
    result = np.zeros_like(orig)
    for i in range(0, len(orig), 5):
        prev = orig.iloc[i]
        for j in range(5):
            if i + j >= len(result):
                break
            result[i + j] = prev * (1 + predictions[i + j])
            prev = result[i + j]
    index = filtered_df[filtered_df.index >= '2024-01-01'].index
    result = pd.DataFrame(result, columns=true_prices.columns, index=index)
    return result


#===================================================================Plot============================================================

def plot_xgb_preds(predictions, true_labels):
    """
    Plot the true labels and predictions for a given ticker.
    
    Arguments:
        X_test (pd.DataFrame): Test feature matrix used for plotting.
        predictions (np.ndarray): Predicted target values.
        true_labels (list): True target values corresponding to the test set.
    
    Returns:
        None: Displays a plot of true labels vs predictions.
    """
    plt.figure(figsize=(12,8))
    index = [i for i in range(len(predictions))]
    
    plt.scatter(
        index,
        true_labels,
        color='pink',
        label='True Labels',
        s=200
    )

    plt.scatter(
        index,
        predictions,
        color='purple',
        label='Predictions',
        marker= 'x',
        s=200
    )

    plt.xlabel('Time [days]')
    plt.ylabel('Close Price')
    plt.title('True Labels vs Predictions')
    plt.axhline(0, color='red', linestyle=":", linewidth=2)  # To add a baseline

    # Displaying the legend
    plt.legend()
    plt.show()


def plot_mean_error(results_df):
    """
    Plot the average mean squared error (MSE) across all tickers.
    
    Arguments:
        results_df (pd.DataFrame): DataFrame containing the results with mean_error.
    
    Returns:
        None
    """
    # Plotting the mean squared error (MSE) across tickers
    plt.figure(figsize=(15, 8))
    plt.bar(results_df['ticker'], results_df['mean_error'], color='pink')
    plt.xlabel('Ticker')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error (MSE) Across Tickers')
    #plt.xticks(rotation=90)
    # Hide x-tick labels
    plt.gca().set_xticklabels([])  # Removes the labels but keeps the ticks
    plt.tight_layout()
    plt.show()

def plot_mean_r2(results_df):
    """
    Plot the average R-squared (R²) across all tickers.
    
    Arguments:
        results_df (pd.DataFrame): DataFrame containing the results with mean_r2.
    
    Returns:
        None
    """
    # Plotting the average R² score across tickers
    plt.figure(figsize=(15, 8))
    plt.bar(results_df['ticker'], results_df['mean_r2'], color='purple')
    plt.xlabel('Ticker')
    plt.ylabel('Mean R-squared (R²)')
    plt.title('Average R-squared (R²) Across Tickers')
    #plt.xticks(rotation=90)
    # Hide x-tick labels
    plt.gca().set_xticklabels([])  # Removes the labels but keeps the ticks
    plt.tight_layout()
    plt.show()


def plot_feature_importance(results_df):
    """
    Plot the feature importances for each ticker.
    
    Arguments:
        results_df (pd.DataFrame): DataFrame containing the results with feature_importances.
    
    Returns:
        None
    """
    # Extract feature importance from the results dataframe
    feature_importances = results_df['feature_importances'].apply(lambda x: np.array(eval(x)))
    
    # Plotting the feature importances for each ticker
    plt.figure(figsize=(12, 8))
    
    for idx, ticker in enumerate(results_df['ticker']):
        importances = feature_importances.iloc[idx]
        plt.bar(np.arange(len(importances)), importances, label=ticker)
    
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importances for Each Ticker')
    plt.legend(title="Tickers", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt

def plot_mse_and_r2_side_by_side(results_df):
    """
    Plot Mean Error and Mean R² side by side in subplots.
    
    Arguments:
        results_df (pd.DataFrame): DataFrame containing the results with mean_error and mean_r2.
    
    Returns:
        None
    """
    # Set up a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

    # Plot 1: Mean Error
    axes[0].bar(results_df['ticker'], results_df['mean_error'], color='lightblue')
    axes[0].set_title('Mean Squared Error (MSE)')
    axes[0].set_xlabel('Ticker')
    axes[0].set_ylabel('Mean Error')
    axes[0].tick_params(axis='x', rotation=90)

    # Plot 2: Mean R²
    axes[1].bar(results_df['ticker'], results_df['mean_r2'], color='salmon')
    axes[1].set_title('Mean R-squared (R²)')
    axes[1].set_xlabel('Ticker')
    axes[1].set_ylabel('Mean R²')
    axes[1].tick_params(axis='x', rotation=90)

    # Show the combined plot
    plt.suptitle('Mean Error and R² Across Tickers', fontsize=16)
    plt.show()

