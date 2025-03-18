import pandas as pd
import numpy as np
from pycaret.time_series import TSForecastingExperiment
from pycaret.regression import RegressionExperiment 


# Load forecast data with 'pm_2_5' column
def load_forecast_data():
    forecast_paths = {
        "jsps001": "data/export-jsps001-1h_processed.csv",  # ข้อมูลที่ไม่มี fe
        "jsps016": "data/export-jsps016-1h_processed.csv",
        "jsps018": "data/export-jsps018-1h_processed.csv",
        "jsps001_fe": "data/export-jsps001-1hfe_processed.csv",  # ข้อมูลที่มี fe
        "jsps016_fe": "data/export-jsps016-1hfe_processed.csv",
        "jsps018_fe": "data/export-jsps018-1hfe_processed.csv",
    }
    forecast_data = {}
    for loc, path in forecast_paths.items():
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.index = df.index.to_period("D")
            forecast_data[loc] = df
            print(f"Loaded forecast data for {loc}: {df.shape} rows and columns")
        except Exception as e:
            print(f"Error loading forecast data for {loc}: {e}")
            forecast_data[loc] = pd.DataFrame()  # Create an empty DataFrame if loading fails
    return forecast_data

# Load forecast data
forecast_data = load_forecast_data()

# Initialize TSForecastingExperiment
exp = TSForecastingExperiment()
exp2 = RegressionExperiment()

# Load pre-trained models using TSForecastingExperiment
models = {
    "arima": {
        "jsps001": exp.load_model("models/export-jsps001-1h"),
        "jsps016": exp.load_model("models/export-jsps016-1h"),
        "jsps018": exp.load_model("models/export-jsps018-1h"),
    },
    "regression": {
        "jsps001": exp.load_model("models/export-jsps001-1hre"),
        "jsps016": exp.load_model("models/export-jsps016-1hre"),
        "jsps018": exp.load_model("models/export-jsps018-1hre"),
    }
}

# Make predictions using ARIMA models
def make_arima_predictions(station_key, days_to_forecast):
    if "arima" not in models or station_key not in models["arima"] or station_key not in forecast_data:
        raise ValueError(f"No ARIMA model or data available for {station_key}")
    
    model = models["arima"][station_key]
    prediction_data = forecast_data[f"{station_key}_fe"]
    
    try:
        # Run setup before prediction
        predictions = exp.predict_model(model, fh=days_to_forecast, X=prediction_data.drop(columns="pm_2_5"))
        predicted_values = predictions["y_pred"]
        future_dates = predictions.index.to_timestamp()
    except Exception as e:
        print(f"Error in ARIMA prediction: {e}")
        # Fallback to simple prediction if error occurs
        last_date = prediction_data.index[-1].to_timestamp()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast, freq='D')
        # Use last value as prediction
        predicted_values = np.array([prediction_data['pm_2_5'].iloc[-1]] * days_to_forecast)
    
    return predicted_values, future_dates

# Make predictions using regression models
def make_regression_predictions(station_key, days_to_forecast):
    if "regression" not in models or station_key not in models["regression"] or station_key not in forecast_data:
        raise ValueError(f"No regression model or data available for {station_key}")
    
    model = models["regression"][station_key]
    prediction_data = forecast_data[f"{station_key}_fe"]
    
    try:
        predictions = exp2.predict_model(model, data=prediction_data.drop(columns="pm_2_5"))
        predicted_values = predictions["prediction_label"]
        future_dates = predictions.index.to_timestamp()
    except Exception as e:
        print(f"Error in regression prediction: {e}")
        # Fallback to simple prediction if error occurs
        last_date = prediction_data.index[-1].to_timestamp()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast, freq='D')
        # Use last value as prediction
        predicted_values = np.array([prediction_data['pm_2_5'].iloc[-1]] * days_to_forecast)
    
    return predicted_values, future_dates

# Make hybrid predictions by combining ARIMA and regression
def make_hybrid_predictions(station_key, days_to_forecast):
    # Get predictions from both models
    arima_values, arima_dates = make_arima_predictions(station_key, days_to_forecast)
    regression_values, regression_dates = make_regression_predictions(station_key, days_to_forecast)
    
    # Calculate hybrid predictions by averaging
    hybrid_values = (arima_values + regression_values) / 2
    
    # Use dates from ARIMA prediction (they should be the same as regression dates)
    future_dates = arima_dates
    
    return hybrid_values, future_dates

