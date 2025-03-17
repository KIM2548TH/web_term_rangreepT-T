from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
import plotly.express as px
import pandas as pd
import numpy as np
from pycaret.time_series import load_model, predict_model
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from math import sqrt
from forecast_utils import make_predictions  # Import the function


# โโละข้อมูลจากไฟล์ CSV
def load_data():
    data_paths = {
        "jsps001_fe": "data/export-jsps001-1hfe_processed.csv",  # ข้อมูลที่มี fe
        "jsps016_fe": "data/export-jsps016-1hfe_processed.csv",
        "jsps018_fe": "data/export-jsps018-1hfe_processed.csv",
        "jsps001": "data/export-jsps001-1h_processed.csv",  # ข้อมูลที่ไม่มี fe
        "jsps016": "data/export-jsps016-1h_processed.csv",
        "jsps018": "data/export-jsps018-1h_processed.csv",
        "jsps001full": "data/export-jsps001-1hfull_processed.csv",  # ข้อมูลที่ไม่มี fe
        "jsps016full": "data/export-jsps016-1hfull_processed.csv",
        "jsps018full": "data/export-jsps018-1hfull_processed.csv",
    }
    data = {}
    for loc, path in data_paths.items():
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if df.empty:
                print(f"Warning: {loc} data is empty!")
            else:
                print(f"Loaded data for {loc}: {df.shape} rows and columns")
            data[loc] = df
        except Exception as e:
            print(f"Error loading data for {loc}: {e}")
            data[loc] = pd.DataFrame()  # สร้าง DataFrame ว่างหากโหลดข้อมูลไม่สำเร็จ
    return data


# โหลดข้อมูล
historical_data = load_data()

# ตรวจสอบว่าข้อมูลถูกโหลดมาจริงหรือไม่
if not historical_data:
    print("Error: historical_data is empty!")
else:
    print("Data loaded successfully!")

# โหลดโมเดล
models = {
    "1D": {
        "jsps001": load_model("models/export-jsps001-1h"),
        "jsps016": load_model("models/export-jsps016-1h"),
        "jsps018": load_model("models/export-jsps018-1h"),
        "jsps001re": load_model("models/export-jsps001re-1h"),
        "jsps016re": load_model("models/export-jsps016-1hre"),
        "jsps018re": load_model("models/export-jsps018-1hre"),
    },
}


# สถานที่และตำแหน่ง
locations = {
    "jsps001": {"name": "JSPs001", "lat": 13.7563, "lon": 100.5018},
    "jsps016": {"name": "JSPs016", "lat": 13.7363, "lon": 100.5218},
    "jsps018": {"name": "JSPs018", "lat": 13.7263, "lon": 100.5318},
}

# สร้าง Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Update the layout to match the weather app design and fit screen
# Update the layout to fit the screen better
app.layout = html.Div(  # Use html.Div for full-screen layout
    [
        dbc.Row(
            dbc.Col(
                html.H1(
                    children="PM2.5 Prediction Dashboard", 
                    className="text-center my-2",
                    style={"fontSize": "2rem", "fontWeight": "bold", "color": "white"}
                )
            ),
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dl.Map(
                            [
                                dl.TileLayer(),
                                *[
                                    dl.Marker(
                                        position=[
                                            locations[loc]["lat"],
                                            locations[loc]["lon"],
                                        ],
                                        children=dl.Tooltip(locations[loc]["name"]),
                                    )
                                    for loc in locations
                                ],
                                dl.LayerGroup(id="map-click-layer"),
                            ],
                            id="map",
                            style={"height": "300px", "width": "100%", "borderRadius": "15px"},
                            center=[13.7363, 100.5218],
                            zoom=12,
                        ),
                        html.Div(className="mt-2", children=[
                            dbc.Card(
                                [
                                    dbc.CardHeader("PM2.5 Prediction"),
                                    dbc.CardBody([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Button("ARIMA", id="predict-arima-button", color="primary", className="me-2", size="sm"),
                                                dbc.Button("Regression", id="predict-regression-button", color="success", className="me-2", size="sm"),
                                                dbc.Button("Hybrid", id="predict-hybrid-button", color="warning", size="sm"),  # Ensure Hybrid button is present
                                            ], width=12, className="mb-2"),
                                        ]),
                                        dcc.Graph(id="prediction-plot", style={"height": "300px"}),  # Increase graph height
                                    ]),
                                ],
                                style={"borderRadius": "15px", "backgroundColor": "#1e1e1e", "color": "white"}
                            ),
                        ]),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {
                                                "label": locations[loc]["name"],
                                                "value": loc,
                                            }
                                            for loc in locations
                                        ],
                                        value="jsps001",
                                        id="station-dropdown",
                                        className="form-control mb-2",
                                        style={"borderRadius": "10px"}
                                    ),
                                    width=12,
                                ),
                            ]
                        ),
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Current PM2.5 Level"),
                                        dbc.CardBody(
                                            [
                                                html.H2(id="current-pm25", className="card-title text-center"),
                                                html.P("μg/m³", className="text-center text-muted"),
                                                html.Div(id="station-info-details")  # Add a div to hold additional info
                                            ]
                                        ),
                                    ],
                                    className="mb-2",
                                    style={"borderRadius": "15px", "backgroundColor": "#1e1e1e", "color": "white"}
                                ),
                                width=12,
                            )
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(id="card-1"),  # Add ID for Card 1
                                        style={"height": "100px", "backgroundColor": "#1e1e1e", "color": "white", "borderRadius": "15px"}
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(id="card-2"),  # Add ID for Card 2
                                        style={"height": "100px", "backgroundColor": "#1e1e1e", "color": "white", "borderRadius": "15px"}
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(id="card-3"),  # Add ID for Card 3
                                        style={"height": "100px", "backgroundColor": "#1e1e1e", "color": "white", "borderRadius": "15px"}
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(id="card-4"),  # Add ID for Card 4
                                        style={"height": "100px", "backgroundColor": "#1e1e1e", "color": "white", "borderRadius": "15px"}
                                    ),
                                    width=3,
                                ),
                            ],
                            className="mb-2"
                        ),
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("7-Day PM2.5 Forecast"),
                                        dbc.CardBody(
                                            [
                                                html.Div(id="forecast-display", style={"height": "400px", "overflowY": "auto"}),  # Increase height
                                            ]
                                        ),
                                    ],
                                    className="mb-2",
                                    style={"borderRadius": "15px", "backgroundColor": "#1e1e1e", "color": "white"}
                                ),
                                width=12,
                            )
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("PM2.5 History"),
                        dbc.CardBody(
                            [
                                dcc.Graph(id="station-plot", style={"height": "500px"}),  # Increase graph height
                            ]
                        ),
                    ],
                    style={"borderRadius": "15px", "backgroundColor": "#1e1e1e", "color": "white"}
                ),
                width=12,
            )
        ),
    ],
    style={"padding": "0", "height": "100vh", "overflow": "auto", "backgroundColor": "#0e0e0e"}  # Make container scrollable
)

# Update callback to display current data and forecast
# Combine callbacks for dashboard update and prediction
@app.callback(
    Output("station-plot", "figure"),
    Output("forecast-display", "children"),
    Output("current-pm25", "children"),
    Output("card-1", "children"),
    Output("card-2", "children"),
    Output("card-3", "children"),
    Output("card-4", "children"),
    Output("prediction-plot", "figure"),
    [Input("station-dropdown", "value"),
     Input("predict-arima-button", "n_clicks"),
     Input("predict-regression-button", "n_clicks"),
     Input("predict-hybrid-button", "n_clicks")]
)
def update_dashboard_and_prediction(selected_station, arima_clicks, regression_clicks, hybrid_clicks):
    ctx = callback_context
    if not selected_station:
        print("No station selected")
        return {}, [], "No Data", "", "", "", "", {}

    try:
        # Get data for the selected station from the full file
        full_file_key = f"{selected_station}full"
        print(f"Attempting to load data from {full_file_key}")

        # Check if the key exists in historical_data
        if full_file_key not in historical_data:
            print(f"Error: {full_file_key} not found in historical_data keys")
            print(f"Available keys: {list(historical_data.keys())}")
            return {}, [], "Data Not Found", "", "", "", "", {}

        station_data_full = historical_data[full_file_key]

        # Check if data is empty
        if station_data_full is None or station_data_full.empty:
            print(f"Error: Data for {full_file_key} is empty or None")
            return {}, [], "No Data Available", "", "", "", "", {}

        print(f"Successfully loaded data for {full_file_key}")
        print(f"Data shape: {station_data_full.shape}")
        print(f"Data columns: {station_data_full.columns.tolist()}")
        print(f"First row: {station_data_full.iloc[0].to_dict()}")
        print(f"Last row: {station_data_full.iloc[-1].to_dict()}")

        # Ensure the data is sorted by timestamp
        station_data_full = station_data_full.sort_values('timestamp')

        # Create the time series plot with all numeric columns
        last_date = station_data_full['timestamp'].max()
        seven_days_ago = last_date - pd.Timedelta(days=7)
        filtered_data = station_data_full[station_data_full['timestamp'] >= seven_days_ago]

        print(f"Filtered data shape: {filtered_data.shape}")

        # Get all numeric columns except timestamp for plotting
        numeric_columns = filtered_data.select_dtypes(include=['number']).columns
        print(f"Numeric columns: {numeric_columns.tolist()}")

        if len(numeric_columns) == 0:
            print("Error: No numeric columns found in data")
            return {}, [], "No Numeric Data", "", "", "", "", {}

        # Create the time series plot with all numeric columns
        fig = px.line(
            filtered_data,
            x="timestamp",
            y=numeric_columns,  # Show all numeric columns
            title=f"All Features for {locations[selected_station]['name']} (Last 7 Days)",
            labels={"value": "Feature Values", "timestamp": "Date"},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Date",
            yaxis_title="Feature Values",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend_title="Features"
        )

        # Get the latest data for station info
        latest_data = station_data_full.iloc[-1]

        # Assign each feature to a card
        card_1_content = f"Temperature: {latest_data.get('temperature', 'N/A')}°C"
        card_2_content = f"Humidity: {latest_data.get('humidity', 'N/A')}%"
        card_3_content = f"PM2.5 SP: {latest_data.get('pm_2_5_sp', 'N/A')} μg/m³"
        card_4_content = f"PM2.5: {latest_data.get('pm_2_5', 'N/A')} μg/m³"

        # Create forecast display using actual data from full file
        days = ["วันนี้", "พรุ่งนี้", "อีก 2 วัน", "อีก 3 วัน", "อีก 4 วัน", "อีก 5 วัน", "อีก 6 วัน"]
        forecast_rows = []

        # Add current day
        current_day = html.Div(
            [
                html.Div("วันนี้", className="font-weight-bold"),
                html.Div(f"{latest_data.get('pm_2_5', 'N/A')} μg/m³"),
            ],
            className="d-flex justify-content-between align-items-center mb-2",
        )
        forecast_rows.append(current_day)

        # Add forecast for next days using actual data
        for i in range(1, 7):  # Next 6 days (total 7 including today)
            if i < len(station_data_full):
                day_data = station_data_full.iloc[-i-1]
                forecast_row = html.Div(
                    [
                        html.Div(days[i-1], className="font-weight-bold"),
                        html.Div(f"{day_data.get('pm_2_5', 'N/A')} μg/m³"),
                    ],
                    className="d-flex justify-content-between align-items-center mb-2",
                )
                forecast_rows.append(forecast_row)

        # Return the current PM2.5 value for the dedicated display
        current_pm25 = f"{latest_data.get('pm_2_5', 'N/A')}"

        # Determine which button was clicked for prediction
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "predict-arima-button":
            model_type = "arima"
            predicted_values, future_dates = make_predictions(selected_station, 7, model_type=model_type)
        elif button_id == "predict-regression-button":
            model_type = "regression"
            predicted_values, future_dates = make_predictions(selected_station, 7, model_type=model_type)
        elif button_id == "predict-hybrid-button":
            # Get predictions from both models
            arima_values, _ = make_predictions(selected_station, 7, model_type="arima")
            regression_values, future_dates = make_predictions(selected_station, 7, model_type="regression")
            # Calculate hybrid predictions by averaging
            predicted_values = (arima_values + regression_values) / 2
        else:
            return fig, forecast_rows, current_pm25, card_1_content, card_2_content, card_3_content, card_4_content, {}

        # Create the prediction plot
        prediction_fig = px.line(
            x=future_dates,
            y=predicted_values,
            title=f"{model_type.capitalize()} Prediction for {locations[selected_station]['name']} (Next 7 Days)",
            labels={"y": "Predicted PM2.5 (μg/m³)", "x": "Date"},
            color_discrete_sequence=['#FF5733']
        )
        
        # Update layout for better visualization
        prediction_fig.update_layout(
            height=200,  # Reduced height to fit screen
            margin=dict(l=20, r=20, t=40, b=20),  # Reduced margins
            xaxis_title="",
            yaxis_title="PM2.5",
            showlegend=False,
            plot_bgcolor='white',  # Set background color to white
            paper_bgcolor='white'  # Set paper background color to white
        )
        
        return fig, forecast_rows, current_pm25, card_1_content, card_2_content, card_3_content, card_4_content, prediction_fig

    except Exception as e:
        print(f"Error updating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return {}, [], "Error", "", "", "", "", {}

if __name__ == "__main__":
    app.run_server(debug=True)
