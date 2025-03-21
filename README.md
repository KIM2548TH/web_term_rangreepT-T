# web_term_rangreepT-T
# PM2.5 Prediction Dashboard

This project is a dashboard for predicting PM2.5 levels at different monitoring stations in Thailand. The dashboard provides visualization of historical data and predictions using different models.

## Features

- Interactive map showing monitoring station locations
- Real-time PM2.5 level display
- Historical data visualization
- Prediction capabilities using multiple models:
  - ARIMA model
  - Regression model
  - Hybrid model (average of ARIMA and Regression)
- Responsive design for various screen sizes

## Technical Details

The dashboard is built using:
- Python with Dash framework
- Plotly for interactive visualizations
- PyCaret for time series forecasting
- Bootstrap for responsive design

## Station Information

The system currently monitors three stations:
- JSPs001
- JSPs016
- JSPs018

Each station provides data on:
- PM2.5 levels
- Temperature
- Humidity
- Other environmental factors

## Prediction Models

### ARIMA Model
Time series forecasting using Auto-Regressive Integrated Moving Average.

### Regression Model
Machine learning regression model trained on historical data.

### Hybrid Model
Combines predictions from both ARIMA and Regression models by averaging their results for potentially more robust predictions.

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access the dashboard at http://127.0.0.1:8050/

## Data Sources

The system uses processed data from monitoring stations, with both raw data and feature-engineered datasets available for different prediction approaches.