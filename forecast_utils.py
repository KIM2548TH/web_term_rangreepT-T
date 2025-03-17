import pandas as pd
from pycaret.time_series import TSForecastingExperiment

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

# Load pre-trained models using TSForecastingExperiment
models = {
    "arima": {
        "jsps001": exp.load_model("models/export-jsps001-1h"),
        "jsps016": exp.load_model("models/export-jsps016-1h"),
        "jsps018": exp.load_model("models/export-jsps018-1h"),
    },
    "regression": {
        "jsps001": exp.load_model("models/export-jsps001re-1h"),
        "jsps016": exp.load_model("models/export-jsps016-1hre"),
        "jsps018": exp.load_model("models/export-jsps018-1hre"),
    }
}

# Make predictions using pre-trained models
def make_predictions(station_key, days_to_forecast, model_type="arima"):
    if model_type not in models or station_key not in models[model_type] or station_key not in forecast_data:
        raise ValueError(f"No model or data available for {station_key} with model type {model_type}")
    
    model = models[model_type][station_key]
    prediction_data = forecast_data[f"{station_key}_fe"]
    
    # Run setup before prediction
    predictions = exp.predict_model(model, fh=7, X=prediction_data.drop(columns="pm_2_5"))
    predicted_values = predictions["y_pred"]
    future_dates = predictions.index.to_timestamp()

    return predicted_values, future_dates

# Make predictions using regression models
def make_regression_predictions(station_key, days_to_forecast):
    if station_key not in models["regression"] or station_key not in forecast_data:
        raise ValueError(f"No regression model or data available for {station_key}")
    
    model = models["regression"][station_key]
    prediction_data = forecast_data[f"{station_key}_fe"]
    
    # Run setup before prediction
    predictions = exp.predict_model(model, fh=days_to_forecast, X=prediction_data.drop(columns="pm_2_5"))
    predicted_values = predictions["y_pred"]
    future_dates = predictions.index.to_timestamp()

    return predicted_values, future_dates

# Example usage for regression
regression_predicted_values, regression_future_dates = make_regression_predictions("jsps001", 7)
