import pandas as pd


def remove_outliers_iqr(df):
    if df is None:
        raise ValueError("DataFrame is None. Please check the input data.")

    if "timestamp" not in df.columns:
        raise ValueError("Column 'timestamp' not found in DataFrame.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df.set_index("timestamp", inplace=True)
    df.drop(
        columns=["timezone", "Unnamed: 0", "location"], inplace=True, errors="ignore"
    )

    columns_to_clean = ["pm_2_5", "temperature", "humidity"]
    columns_to_clean = [col for col in columns_to_clean if col in df.columns]

    for col in columns_to_clean:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    df.interpolate(method="linear", inplace=True)
    df = df.resample("D").mean().fillna(method="ffill")
    df.index = df.index.to_period("D")
    return df


def add_lag_features(df):
    lags = [8, 10, 14, 21]
    for lag in lags:
        for col in df.columns:
            if col in df.columns:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df, shift=7):
    windows = [2, 3, 5, 7, 14]
    for window in windows:
        for col in df.columns:
            if col in df.columns:
                df[f"{col}_rollmean{window}"] = (
                    df[col].shift(shift).rolling(window=window, min_periods=1).mean()
                )
                df[f"{col}_rollstd{window}"] = (
                    df[col].shift(shift).rolling(window=window, min_periods=1).std()
                )
    return df


def preprocess_data(df):
    expected_columns = ["temperature", "humidity", "pm_2_5_sp", "pm_10"]
    for col in expected_columns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    one_year_ago = df.index.max() - pd.DateOffset(years=2)
    df = df[df.index >= one_year_ago]
    df = df.asfreq("D").fillna(method="bfill")
    return df


def prepare_forecast_features(df, forecast_days=8):
    X_forecast = df.iloc[-forecast_days:].copy()
    selected_features = ["pm_2_5"] + [col for col in df.columns if col != "pm_2_5"]
    X_forecast = X_forecast[selected_features]
    X_forecast.fillna(method="ffill", inplace=True)
    return X_forecast
