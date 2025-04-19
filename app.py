import os
import pandas as pd
import holidays
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table
import numpy as np



GADB_COLS = [
    "icao_code", "iata_code", "airport_name", "city", "country",
    "lat_deg", "lat_min", "lat_sec", "lat_dir",
    "lon_deg", "lon_min", "lon_sec", "lon_dir",
    "altitude", "lat_dec", "lon_dec"
]

coords_df  = pd.read_csv(
    "data/GlobalAirportDatabase.txt",
    sep=":", header=None, names=GADB_COLS
)

usa_airports = coords_df.loc[coords_df["country"] == "USA",
                             ["iata_code", "airport_name", "city",
                              "lat_dec", "lon_dec"]]

usa_airports.rename(columns={"lat_dec": "lat", "lon_dec": "lon"},
                    inplace=True)

airport_coords = {
    "LAX": (33.9416, -118.4085),          # Los Angeles (origin!)
    "ABQ": (35.0402, -106.6090),          # Albuquerque International Sunport
    "ANC": (61.1744, -149.9964),          # Ted Stevens Anchorage International Airport
    "ATL": (33.6407, -84.4277),           # Hartsfield–Jackson Atlanta International Airport
    "AUS": (30.1944, -97.6699),           # Austin-Bergstrom International Airport
    "BDL": (41.9389, -72.6832),           # Bradley International Airport (Hartford, CT)
    "BHM": (33.5636, -86.7533),           # Birmingham-Shuttlesworth International Airport
    "BNA": (36.1244, -86.6783),           # Nashville International Airport
    "BOI": (43.5646, -116.2225),          # Boise Airport
    "BOS": (42.3656, -71.0096),           # Boston Logan International Airport
    "BTR": (30.5334, -91.1549),           # Baton Rouge Metropolitan Airport
    "BUF": (42.9405, -78.7322),           # Buffalo Niagara International Airport
    "BWI": (39.1754, -76.6684),           # Baltimore/Washington International Thurgood Marshall Airport
    "BZN": (45.7772, -111.1522),          # Bozeman Yellowstone International Airport
    "CHS": (32.8986, -80.0405),           # Charleston International Airport
    "CID": (41.8847, -91.7103),           # Eastern Iowa Airport (Cedar Rapids)
    "CLE": (41.4112, -81.8498),           # Cleveland Hopkins International Airport
    "CLT": (35.2144, -80.9431),           # Charlotte Douglas International Airport
    "CMH": (39.9980, -82.8919),           # John Glenn Columbus International Airport
    "CVG": (39.0489, -84.6675),           # Cincinnati/Northern Kentucky International Airport
    "DAL": (32.8471, -96.8517),           # Dallas Love Field
    "DCA": (38.8512, -77.0402),           # Ronald Reagan Washington National Airport
    "DEN": (39.8561, -104.6737),          # Denver International Airport
    "DFW": (32.8998, -97.0403),           # Dallas/Fort Worth International Airport
    "DSM": (41.5348, -93.6637),           # Des Moines International Airport
    "DTW": (42.2124, -83.3534),           # Detroit Metropolitan Wayne County Airport
    "EGE": (39.6426, -106.9170),          # Eagle County Regional Airport
    "ELP": (31.8070, -106.3770),          # El Paso International Airport
    "EUG": (44.1240, -123.2110),          # Eugene Airport
    "EWR": (40.6895, -74.1745),           # Newark Liberty International Airport
    "FLL": (26.0726, -80.1527),           # Fort Lauderdale-Hollywood International Airport
    "GEG": (47.6219, -117.5333),          # Spokane International Airport
    "HNL": (21.3187, -157.9225),          # Daniel K. Inouye International Airport (Honolulu)
    "HOU": (29.6454, -95.2789),           # William P. Hobby Airport (Houston)
    "IAD": (38.9531, -77.4565),           # Washington Dulles International Airport
    "IAH": (29.9844, -95.3414),           # George Bush Intercontinental Airport (Houston)
    "IND": (39.7173, -86.2944),           # Indianapolis International Airport
    "ITO": (19.7297, -155.0896),          # Hilo International Airport
    "JAC": (43.6055, -110.7377),          # Jackson Hole Airport
    "JAX": (30.4941, -81.6879),           # Jacksonville International Airport
    "JFK": (40.6413, -73.7781),           # John F. Kennedy International Airport
    "KOA": (19.7388, -156.0447),          # Kona International Airport at Keahole
    "LAS": (36.0840, -115.1537),          # McCarran International Airport (Las Vegas)
    "LGA": (40.7769, -73.8740),           # LaGuardia Airport
    "LIH": (21.9750, -159.3381),          # Lihue Airport (Hawaii)
    "MCI": (39.2976, -94.7139),           # Kansas City International Airport
    "MCO": (28.4312, -81.3081),           # Orlando International Airport
    "MDW": (41.7868, -87.7522),           # Chicago Midway International Airport
    "MEM": (35.0424, -89.9767),           # Memphis International Airport
    "MIA": (25.7959, -80.2870),           # Miami International Airport
    "MKE": (42.9472, -87.8962),           # General Mitchell International Airport (Milwaukee)
    "MSN": (43.1399, -89.3370),           # Dane County Regional Airport (Madison)
    "MSP": (44.8848, -93.2223),           # Minneapolis–Saint Paul International Airport
    "MSY": (29.9934, -90.2580),           # Louis Armstrong New Orleans International Airport
    "MTJ": (38.5167, -107.8833),          # Montrose Regional Airport
    "OAK": (37.7126, -122.2197),          # Oakland International Airport
    "OGG": (20.8966, -156.4305),          # Kahului Airport (Maui)
    "OMA": (41.3030, -96.1244),           # Eppley Airfield (Omaha)
    "ORD": (41.9742, -87.9073),           # O'Hare International Airport (Chicago)
    "PAE": (47.9060, -122.2816),          # Paine Field (Everett, WA)
    "PBI": (26.6837, -80.0956),           # West Palm Beach International Airport
    "PDX": (45.5898, -122.5951),          # Portland International Airport
    "PHL": (39.8744, -75.2424),           # Philadelphia International Airport
    "PHX": (33.4342, -112.0116),          # Phoenix Sky Harbor International Airport
    "PIT": (40.4915, -80.2329),           # Pittsburgh International Airport
    "PSP": (33.8296, -116.5070),          # Palm Springs International Airport
    "RDM": (44.2540, -121.1500),          # Roberts Field (Redmond, OR)
    "RDU": (35.8776, -78.7875),           # Raleigh-Durham International Airport
    "RIC": (37.5052, -77.3193),           # Richmond International Airport
    "RNO": (39.4993, -119.7681),          # Reno-Tahoe International Airport
    "RSW": (26.5362, -81.7550),           # Southwest Florida International Airport
    "SAN": (32.7338, -117.1933),          # San Diego International Airport
    "SAT": (29.5312, -98.2783),           # San Antonio International Airport
    "SBN": (41.7086, -86.3176),           # South Bend International Airport
    "SDF": (38.1744, -85.7364),           # Louisville Muhammad Ali International Airport
    "SEA": (47.4502, -122.3088),          # Seattle-Tacoma International Airport
    "SFO": (37.6213, -122.3790),          # San Francisco International Airport
    "SJC": (37.3626, -121.9290),          # Norman Y. Mineta San Jose International Airport
    "SLC": (40.7899, -111.9791),          # Salt Lake City International Airport
    "SMF": (38.6951, -121.5915),          # Sacramento International Airport
    "SNA": (33.6757, -117.8678),          # John Wayne Airport (Santa Ana)
    "STL": (38.7487, -90.3700),           # St. Louis Lambert International Airport
    "TPA": (27.9755, -82.5332),           # Tampa International Airport
    "TUL": (36.1984, -95.8881),           # Tulsa International Airport
    "TUS": (32.1161, -110.9410)           # Tucson International Airport
}
try:
    encoder = OneHotEncoder(drop=None,
                            handle_unknown="ignore",
                            sparse_output=False)   
except TypeError:                                   
    encoder = OneHotEncoder(drop=None,
                            handle_unknown="ignore",
                            sparse=False)        
    
def build_airline_map(carrier_code: str, user_dest: str | None = None) -> go.Figure:
    """Return a ScatterGeo map for carrier routes 2021‑24."""
    df = data[
        (data["Carrier Code"] == carrier_code) & (data["year"].between(2021, 2024))
    ].copy()
    if df.empty:
        return go.Figure()

    df = (
        df.merge(usa_airports, how="left", left_on="destination_airport", right_on="iata_code")
          .assign(origin_lat=33.9416, origin_lon=-118.4085)
    )

    summary = (
        df.groupby("destination_airport")
          .agg(lat=("lat", "first"),
               lon=("lon", "first"),
               delay_p=("delayed", "mean"),
               flights=("delayed", "size"))
          .reset_index()
    )

    fig = go.Figure()

    # flight paths
    for _, r in summary.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[df["origin_lon"].iat[0], r.lon],
            lat=[df["origin_lat"].iat[0], r.lat],
            mode="lines",
            line=dict(width=1, color="royalblue"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # all US airports
    fig.add_trace(go.Scattergeo(
        lon=usa_airports["lon"], lat=usa_airports["lat"],
        mode="markers", marker=dict(size=5, color="steelblue", opacity=0.25),
        hoverinfo="skip", name="U.S. airports"
    ))

    # destinations
    fig.add_trace(go.Scattergeo(
        lon=summary["lon"], lat=summary["lat"], mode="markers",
        marker=dict(
            size=8 + summary["flights"] / summary["flights"].max() * 18,
            color=summary["delay_p"], colorscale="RdYlGn_r",
            showscale=True, cmin=0, cmax=1,
            colorbar=dict(title="Delay Proportion", x=1.01, len=0.75),
        ),
        text=(summary["destination_airport"] + "<br>" +
              (summary["delay_p"]*100).round(1).astype(str) + "% delayed<br>" +
              summary["flights"].astype(str) + " flights"),
        hoverinfo="text", name="Destinations",
    ))

    # highlighted destination (yellow ring)
    if user_dest in summary["destination_airport"].values:
        sel = summary.loc[summary["destination_airport"].eq(user_dest)].iloc[0]
        fig.add_trace(go.Scattergeo(
            lon=[sel.lon], lat=[sel.lat], mode="markers",
            marker=dict(size=8 + sel.flights / summary["flights"].max() * 18 + 6,
                        color="rgba(0,0,0,0)", line=dict(width=3, color="#FFD600")),
            hoverinfo="skip", showlegend=False,
        ))
    elif user_dest in airport_coords:  
        d_lat, d_lon = airport_coords[user_dest]
        fig.add_trace(go.Scattergeo(lon=[-118.4085, d_lon], lat=[33.9416, d_lat],
                                    mode="lines", line=dict(width=1, color="grey", dash="dot"),
                                    hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattergeo(lon=[d_lon], lat=[d_lat], mode="markers",
                                    marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2, color="grey")),
                                    text=[f"{user_dest}<br>(no {carrier_code} flights 21‑24)"],
                                    hoverinfo="text", showlegend=False))

    # LAX star
    fig.add_trace(go.Scattergeo(lon=[-118.4085], lat=[33.9416], mode="markers",
                                marker=dict(size=18, color="red", symbol="star"),
                                text=["LAX"], name="LAX"))

    fig.update_layout(
        title_text=f"Flight Paths {carrier_code} from LAX (2021‑24)",
        font=dict(color="white"),
        geo=dict(scope="north america", projection_type="equirectangular",
                  showland=True, landcolor="#1e1e1e", countrycolor="#525252",
                  lonaxis=dict(range=[-170, -60]), lataxis=dict(range=[20, 75])),
        margin=dict(l=0, r=0, t=40, b=70), height=650,
        legend=dict(bgcolor="rgba(0,0,0,0)", y=0.99, x=0.01, font=dict(color="white")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_coloraxes(colorbar_orientation="h", colorbar_y=-0.20, colorbar_x=0.50,
                         colorbar_len=0.60, colorbar_title_side="bottom")
    return fig


# Instantiate the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)

# Setup Logging Directory and Log Path
logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

log_path = os.path.join(logs_dir, "flight_delay_predictions_log.csv")
if not os.path.exists(log_path):
    pd.DataFrame(columns=[
        'timestamp', 'Carrier Code', 'destination_airport', 'scheduled_dep_hour',
        'month', 'day', 'scheduled_elapsed', 'is_holiday', 'is_weekday',
        'season', 'temperature_2m_mean (°C)', 'rain_sum (mm)',
        'precipitation_sum (mm)', 'wind_speed_10m_max (km/h)', 'probability_of_delay'
    ]).to_csv(log_path, index=False)

# Load data
AA = pd.read_csv('data/AA.csv')
AS = pd.read_csv('data/AS.csv')
DL = pd.read_csv('data/DL.csv')
UA = pd.read_csv('data/UA.csv')
WN = pd.read_csv('data/WN.csv')
SA = pd.read_csv('data/SA.csv')
JB = pd.read_csv('data/JB.csv')
la_weather = pd.read_csv('data/la_weather.csv', header=2)

data = pd.concat([AA, AS, DL, UA, WN, SA, JB], ignore_index=True)

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

data['season'] = data['month'].apply(month_to_season)
la_weather.rename(columns={'time': 'flight_date'}, inplace=True)
data = pd.merge(data, la_weather, on='flight_date', how='left')

data['delayed'] = data['dep_delay'].apply(lambda x: 1 if x >= 15 else 0)
us_holidays = holidays.US()
data["flight_date_pd"] = pd.to_datetime(data["flight_date"])
data["weekday"] = data["flight_date_pd"].dt.day_name()
data["is_weekday"] = (data["flight_date_pd"].dt.weekday < 5).astype(int)
data["is_holiday"] = data["flight_date_pd"].apply(lambda x: 1 if x in us_holidays else 0)
data["year"] = data["flight_date_pd"].dt.year

# Feature Selection & One-Hot Encoding

features = [
    'Carrier Code', 'destination_airport', 'scheduled_dep_hour',
    'month', 'day', 'scheduled_elapsed', 'is_holiday', 'is_weekday',
    'season', 'temperature_2m_mean (°C)', 'rain_sum (mm)',
    'precipitation_sum (mm)', 'wind_speed_10m_max (km/h)'
]

X = data[features]
y = data['delayed']

encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
encoded_cols = encoder.fit_transform(
                  X[['Carrier Code', 'destination_airport', 'season']]
              )
if not isinstance(encoded_cols, np.ndarray):
    encoded_cols = encoded_cols.toarray()

encoded_df = pd.DataFrame(
    encoded_cols,
    columns=encoder.get_feature_names_out(
                ['Carrier Code', 'destination_airport', 'season'])
)

X_encoded = pd.concat([X, encoded_df], axis=1).drop(columns=['Carrier Code', 'destination_airport', 'season'])
X_encoded = X_encoded.reset_index(drop=True)
y = y.reset_index(drop=True)

# Balance & Split the Dataset
delayed_mask = y == 1
X_delayed = X_encoded[delayed_mask]
y_delayed = y[delayed_mask]
X_not_delayed = X_encoded[~delayed_mask]
y_not_delayed = y[~delayed_mask]

n_samples = min(len(X_delayed), len(X_not_delayed))
X_delayed_sampled = X_delayed.sample(n=n_samples, random_state=42)
y_delayed_sampled = y_delayed.loc[X_delayed_sampled.index]
X_not_delayed_sampled = X_not_delayed.sample(n=n_samples, random_state=42)
y_not_delayed_sampled = y_not_delayed.loc[X_not_delayed_sampled.index]

X_balanced = pd.concat([X_delayed_sampled, X_not_delayed_sampled], axis=0)
y_balanced = pd.concat([y_delayed_sampled, y_not_delayed_sampled], axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)
features_encoded = X_balanced.columns

# Train the XGBoost Model
xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
y_pred = xgb_classifier.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
importances = xgb_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features_encoded, 'Importance': importances})
print("Feature Importances:\n", feature_importance_df.sort_values(by='Importance', ascending=False))

# Build Carrier Logo Buttons Dynamically
def approximate_zoom(lax_lat, lax_lon, dest_lat, dest_lon):
    """
    Returns a rough zoom level to ensure the entire path is visible.
    The bigger the distance, the lower the zoom.
    """
    lat_dist = abs(dest_lat - lax_lat)
    lon_dist = abs(dest_lon - lax_lon)
    max_dist = max(lat_dist, lon_dist)

    if max_dist < 4:
        return 6   
    elif max_dist < 8:
        return 5
    elif max_dist < 15:
        return 4
    elif max_dist < 25:
        return 3
    else:
        return 2 

unique_carriers = sorted(data["Carrier Code"].dropna().unique())
first_row_carriers = ["AA", "DL", "UA"]  
second_row_carriers = ["WN", "AS", "B6"]  
third_row_carriers = ["NK"]

unique_destinations = sorted(data["destination_airport"].dropna().unique())
print(unique_destinations)

displayed_carriers = first_row_carriers + second_row_carriers + third_row_carriers

def create_carrier_button(carrier):
    return dbc.Button(
        html.Img(
            src=f"/assets/{carrier}.png",
            style={
                "maxWidth": "80%",
                "maxHeight": "80%",
                "objectFit": "contain"
            }
        ),
        id={"type": "carrier-button", "index": carrier},
        color="link",
        style={
            "width": "250px",
            "height": "150px",
            "backgroundColor": "#616161",  
            "border": "none",
            "margin": "0.5rem",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center"
        }
    )

def create_dest_button(airport_code):
    return dbc.Button(
        f"{airport_code}",
        id={"type": "dest-button", "index": airport_code},
        color="link",
        style={
            "width": "200px",
            "height": "60px",
            "backgroundColor": "#616161",
            "border": "none",
            "margin": "0.5rem",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center"
        }
    )


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# Build buttons
def build_carrier_rows(carriers, buttons_per_row):
    rows = []
    for group in chunk(carriers, buttons_per_row):
        row_buttons = [create_carrier_button(c) for c in group]
        rows.append(
            dbc.Row(
                [dbc.Col(btn, width="auto") for btn in row_buttons],
                justify="center",
                className="mb-2"
            )
        )
    return rows

def build_destination_rows(dest_list, buttons_per_row=3):
    rows = []
    for group in chunk(dest_list, buttons_per_row):
        row_buttons = [create_dest_button(dest) for dest in group]
        rows.append(
            dbc.Row(
                [dbc.Col(btn, width="auto") for btn in row_buttons],
                justify="center",
                className="mb-2"
            )
        )
    return rows

destination_rows = build_destination_rows(unique_destinations, 8)

carrier_rows = build_carrier_rows(displayed_carriers, 3)

# Define Helper Function for Inputs
def labeled_input(label, input_component):
    return dbc.Col(
        [
            dbc.Label(label, className="fw-bold mb-1"),
            input_component
        ],
        width=6,
        className="mb-3"
    )

# Multi-Page Wizard: Page Layouts

# Page 0
page0_layout = dbc.Container(
    className="vh-100 d-flex flex-column justify-content-center align-items-center text-center",
    fluid=True,
    children=[
        html.H1("Welcome to the LAX Flight Delay Predictor!", className="mb-4"),
        html.P(
            "Use this website to calculate the probability that your upcoming flight is delayed.",
            className="mb-4"
        ),
        dcc.Link("Next: Select Airline", href="/page1", className="btn btn-primary")
    ]
)

# Page 1
page1_layout = dbc.Container([
    html.H2("Step 1: Select Your Airline", className="text-center my-3"),
    *carrier_rows, 
    html.Br(),
    dbc.Row([
        dbc.Col(
            dcc.Link("Back: Welcome", href="/page0", className="btn btn-secondary"),
            width={"size": 4, "offset": 2}
        ),
        dbc.Col(
            dcc.Link("Next: Enter Flight Details", href="/page2", className="btn btn-primary"),
            width={"size": 4, "offset": 2}
        )
    ])
], fluid=True)

# Page 2
page2_layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H2("Step 2: Enter Destination", className="text-center my-3"),
            width=12
        )
    ], justify="center"),

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dcc.Graph(
                        id='lax-dest-map',
                        style={
                            "height": "600px",  
                            "width":  "1000px"  
                        }
                    )
                ]),
                style={
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"
                }
            ),
            md=9, 
            className="d-flex justify-content-center mb-3"
        ),

        dbc.Col([
            dbc.Label("Destination Airport", className="mb-2 text-center"),
            dcc.Dropdown(
                id='dest',
                options=[
                    {'label': str(d), 'value': str(d)}
                    for d in sorted(data["destination_airport"].dropna().unique())
                ],
                value=(
                    sorted(data["destination_airport"].dropna().unique())[0]
                    if data["destination_airport"].dropna().any() else None
                ),
                style={
                    "width": "200px",
                    "backgroundColor": "#616161"
                },
                className="mb-3"
            )
        ],
            md=3,
            className="d-flex flex-column align-items-center justify-content-center"
        )
    ], justify="center"),

    dbc.Row([
        dbc.Col(
            dcc.Link("Back: Select Airline", href="/page1", className="btn btn-secondary"),
            width={"size": 4},
            className="text-center"
        ),
        dbc.Col(
            dcc.Link("Next: Enter More Flight Details", href="/page3", className="btn btn-primary"),
            width={"size": 4},
            className="text-center"
        )
    ], justify="center", className="my-3")

], fluid=True)



# Page 3
temp_slider   = dcc.Slider(id="temp",   min=-20, max=50,  step=1, value=20,
                           marks={-20:"-20",0:"0",20:"20",40:"40",50:"50"},
                           tooltip={"always_visible":True},
                           className="orange-slider")
rain_slider   = dcc.Slider(id="rain",   min=0,   max=60,  step=1, value=0,
                           marks={0:"0",20:"20",40:"40",60:"60"},
                           tooltip={"always_visible":True},
                           className="blue-slider")
precip_slider = dcc.Slider(id="precip", min=0,   max=60,  step=1, value=0,
                           marks={0:"0",20:"20",40:"40",60:"60"},
                           tooltip={"always_visible":True},
                           className="purple-slider")
wind_slider   = dcc.Slider(id="wind",   min=0,   max=120, step=1, value=10,
                           marks={0:"0",40:"40",80:"80",120:"120"},
                           tooltip={"always_visible":True},
                           className="teal-slider")


page3_layout = dbc.Container(
    fluid=True,
    children=[
        html.H2("Step 3: Enter Other Flight Details",
                className="text-center my-4"),

        dbc.Col(md=8, className="mx-auto", children=[

            dbc.Row([
                dbc.Col([
                    dbc.Label("Flight Date", className="fw-bold"),
                    dcc.DatePickerSingle(
                        id="flight_date",
                        date=datetime.today().strftime("%Y-%m-%d"),
                        display_format="MMM D, YYYY",
                        first_day_of_week=1
                    )
                ], md=4),

                dbc.Col([
                    dbc.Label("Scheduled Departure Time", className="fw-bold"),
                    dbc.Input(id="dep_time", type="time",
                              value="12:00", step=1)
                ], md=4),

                dbc.Col([
                    dbc.Label("Flight Duration (min)", className="fw-bold"),
                    dbc.Input(id="elapsed", type="number",
                              value=180, min=30, max=1000)
                ], md=4)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Checklist(
                    options=[{"label": " Holiday?", "value": 1}],
                    value=[], id="holiday", switch=True
                ), md=6),
                dbc.Col(dbc.Checklist(
                    options=[{"label": " Week‑day?", "value": 1}],
                    value=[1], id="weekday", switch=True
                ), md=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Label("Temperature (°C)", className="fw-bold"),
                temp_slider
            ], className="mb-4"),

            dbc.Row([
                dbc.Label("Rain (mm)", className="fw-bold"),
                rain_slider
            ], className="mb-4"),

            dbc.Row([
                dbc.Label("Precipitation (mm)", className="fw-bold"),
                precip_slider
            ], className="mb-4"),

            dbc.Row([
                dbc.Label("Wind Speed (km/h)", className="fw-bold"),
                wind_slider
            ], className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(dcc.Link("← Back", href="/page2",
                             className="btn btn-secondary w-100"), md=4),
            dbc.Col(dcc.Link("Next →", href="/page4",
                             className="btn btn-warning w-100"),
                    md=4, className="offset-md-4")
        ], className="my-4")
    ]
)




# Page 4
page4_layout = dbc.Container(
    fluid=True,
    children=[

        html.H2("Step 4: Flight Prediction",
                className="text-center my-3"),

        dbc.Row(
    [
        dbc.Col(
            dbc.Button("Predict Delay", id="submit-btn",
                       n_clicks=0, color="warning"),
            width="auto"
        ),

        dbc.Col(                                 
            dbc.Button("Edit inputs", id="edit-btn",
                       n_clicks=0, color="secondary"),
            width="auto"
        ),

        dbc.Col(
            html.Span("(green = edit mode | grey = view mode)",
                      style={"color":"#fff",
                             "fontSize":"0.8rem",
                             "marginLeft":"0.4rem"}),
            width="auto", className="d-flex align-items-center"
        ),
    ],
    className="mb-3 g-2"
)
,

        html.Div(id="prediction-output"),
        html.A(
    "← Back",
    href="/page3",
    className="btn btn-secondary",
    style={
        "position": "fixed",
        "left":     "20px",
        "bottom":   "20px",
        "zIndex":   1020
    }
),

        
    ]
)



# Main Multi-Page Layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="selected-carrier-store", data=unique_carriers[0] if unique_carriers else None),
    dcc.Store(id="flight-details-store"),
    dcc.Store(id="edit-mode-store", data=False),  
    html.Div(id="page-content"),
    html.Div(id="dummy-output", style={"display": "none"})
])

# Navigation 






@app.callback(
    Output("lax-dest-map", "figure"),
    Input("dest", "value")
)
def update_map(selected_dest):
    if not selected_dest or selected_dest not in airport_coords or "LAX" not in airport_coords:
        fig = go.Figure()
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center={"lat": 39.8283, "lon": -98.5795},
                zoom=3
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig

    lax_lat, lax_lon = airport_coords["LAX"]
    dest_lat, dest_lon = airport_coords[selected_dest]

    zoom = approximate_zoom(lax_lat, lax_lon, dest_lat, dest_lon)

    fig = go.Figure()

    all_codes = list(airport_coords.keys())
    all_lats = [airport_coords[c][0] for c in all_codes]
    all_lons = [airport_coords[c][1] for c in all_codes]

    fig.add_trace(go.Scattermapbox(
        lat=all_lats,
        lon=all_lons,
        mode="markers",
        marker=dict(size=5, color="gray"),
        text=all_codes,
        hoverinfo="text",
        name="All Destinations"
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[lax_lat, dest_lat],
        lon=[lax_lon, dest_lon],
        mode="markers",
        marker=dict(size=9, color="red"),
        text=["LAX", selected_dest],
        hoverinfo="text",
        name="Airports"
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[lax_lat, dest_lat],
        lon=[lax_lon, dest_lon],
        mode="lines",
        line=dict(width=4, color="yellow"),
        name="Flight Path"
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": (lax_lat + dest_lat) / 2, "lon": (lax_lon + dest_lon) / 2},
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig





@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname in ["/", "/page0"]:
        return page0_layout
    elif pathname == "/page1":
        return page1_layout
    elif pathname == "/page2":
        return page2_layout
    elif pathname == "/page3":
        return page3_layout
    elif pathname == "/page4":
        return page4_layout
    else:
        return "404: Page Not Found"

# Callback to Highlight Selected Carrier
@app.callback(
    Output({"type": "carrier-button", "index": ALL}, "style"),
    Input("selected-carrier-store", "data")
)
def highlight_selected_carrier(selected_carrier):
    base_style = {
        "width": "250px",
        "height": "150px",
        "backgroundColor": "#616161",  
        "border": "none",
        "margin": "0.5rem",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center"
    }
    highlight_style = {
        "width": "250px",
        "height": "150px",
        "backgroundColor": "#FFC107",  
        "border": "2px solid #2196f3",
        "margin": "0.5rem",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center"
    }
    if not selected_carrier:
        return [base_style] * len(displayed_carriers)
    return [highlight_style if carrier == selected_carrier else base_style for carrier in displayed_carriers]

# Callback to Capture Selected Airline
@app.callback(
    Output("selected-carrier-store", "data"),
    Input({"type": "carrier-button", "index": ALL}, "n_clicks"),
    State({"type": "carrier-button", "index": ALL}, "id"),
    prevent_initial_call=True
)
def update_selected_carrier(n_clicks_list, id_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    selected = eval(triggered_id)["index"]
    return selected

@app.callback(
    Output("flight-details-store", "data", allow_duplicate=True), 
    Input({"type": "dest-button", "index": ALL}, "n_clicks"),
    State({"type": "dest-button", "index": ALL}, "id"),
    State("flight-details-store", "data"),
    prevent_initial_call=True
)
def capture_destination(n_clicks_list, id_list, current_data):
    if not dash.callback_context.triggered or all(x is None for x in n_clicks_list):
        raise dash.exceptions.PreventUpdate
    
    # Identify which button was clicked
    triggered_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    triggered_index = eval(triggered_id)["index"]
    
    if current_data is None:
        current_data = {}
    current_data["dest"] = triggered_index 
    return current_data

@app.callback(
    Output({"type": "dest-button", "index": ALL}, "style"),
    Input("flight-details-store", "data")
)
def highlight_chosen_dest(flight_data):
    base_style = {
        "width": "200px",
        "height": "60px",
        "backgroundColor": "#616161",
        "border": "none",
        "margin": "0.5rem",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center"
    }
    highlight_style = {
        "width": "200px",
        "height": "60px",
        "backgroundColor": "#FFC107",
        "border": "2px solid #2196f3",
        "margin": "0.5rem",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center"
    }

    if not flight_data or "dest" not in flight_data:
        return [base_style] * len(unique_destinations)
    
    chosen_dest = flight_data["dest"]
    styles = []
    for airport_code in unique_destinations:
        if airport_code == chosen_dest:
            styles.append(highlight_style)
        else:
            styles.append(base_style)
    return styles



# Callback to Capture Flight Details from Page 2 and Page 3
@app.callback(
    Output("flight-details-store", "data", allow_duplicate=True),
    Input('dest', 'value'),
    State("flight-details-store", "data"),
    prevent_initial_call=True
)
def capture_destination(dest, current_data):
    if current_data is None:
        current_data = {}
    current_data["dest"] = dest
    return current_data

@app.callback(
    Output("flight-details-store", "data", allow_duplicate=True),
    Input("flip-table", "data"),                     
    State("flight-details-store", "data"),
    prevent_initial_call=True
)
def table_to_store(rows, current):
    if not rows:                       
        raise dash.exceptions.PreventUpdate
    row = rows[0]                     
    store = (current or {}).copy()
    

    # simple field mapping
    store.update({
    "dest":     row.get("dest") or "",        
    "dep_hour": to_float(row.get("dep"),      0),
    "month":    to_int  (row.get("month"),    1),
    "day":      to_int  (row.get("day"),      1),
    "elapsed":  to_int  (row.get("duration"), 180),
    "holiday":  to_int  (row.get("holiday"),  0),
    "weekday":  to_int  (row.get("weekday"),  0),
    "temp":     to_float(row.get("temp"),    20),
    "rain":     to_float(row.get("rain"),     0),
    "precip":   to_float(row.get("precip"),   0),
    "wind":     to_float(row.get("wind"),    10),
})
    return store

@app.callback(
    Output("flight-details-store", "data", allow_duplicate=True),
    [
        Input("dep_time",    "value"),
        Input("flight_date", "date"),
        Input("elapsed",     "value"),
        Input("holiday",     "value"),  
        Input("weekday",     "value"),   
        Input("temp",        "value"),
        Input("rain",        "value"),
        Input("precip",      "value"),
        Input("wind",        "value")
    ],
    State("flight-details-store", "data"),
    prevent_initial_call=True
)
def capture_other_details(dep_time, flight_date_iso,
                          elapsed, holiday_bool, weekday_bool,
                          temp, rain, precip, wind,
                          current_data):
    current_data = current_data or {}

    # departure hour as decimal
    if dep_time:
        h, m, *rest = map(int, dep_time.split(":"))
        s = rest[0] if rest else 0
        current_data["dep_hour"] = h + m/60 + s/3600

    # split date into month / day
    if flight_date_iso:
        dt = datetime.fromisoformat(flight_date_iso)
        current_data["month"] = dt.month
        current_data["day"]   = dt.day

    current_data["elapsed"]  = elapsed
    current_data["holiday"]  = int(bool(holiday_bool))  
    current_data["weekday"]  = int(bool(weekday_bool))
    current_data["temp"]     = temp
    current_data["rain"]     = rain
    current_data["precip"]   = precip
    current_data["wind"]     = wind

    return current_data

def to_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default        

def to_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def capture_flight_details(dest, dep_hour, month, day, elapsed,
                           holiday, weekday, temp, rain, precip, wind,
                           current_data):
    if current_data is None:
        current_data = {}
    if dest is not None:
        current_data["dest"] = dest
    if dep_hour is not None:
        current_data["dep_hour"] = dep_hour
    if month is not None:
        current_data["month"] = month
    if day is not None:
        current_data["day"] = day
    if elapsed is not None:
        current_data["elapsed"] = elapsed
    if holiday is not None:
        current_data["holiday"] = holiday
    if weekday is not None:
        current_data["weekday"] = weekday
    if temp is not None:
        current_data["temp"] = temp
    if rain is not None:
        current_data["rain"] = rain
    if precip is not None:
        current_data["precip"] = precip
    if wind is not None:
        current_data["wind"] = wind
    return current_data

# Final Prediction


@app.callback(
    Output("prediction-output", "children"),
    [Input("submit-btn", "n_clicks"), Input("flight-details-store", "data")],
    State("selected-carrier-store", "data"),
    prevent_initial_call=True,
)
def update_prediction(n_clicks, flight_data, selected_carrier):
    if n_clicks == 0:
        return ""

    flight_data = flight_data or {}

    # ensure dep_hour 
    if "dep_hour" not in flight_data:
        time_str = flight_data.get("dep_time") or "12:00"   
        h, m = map(int, time_str.split(":"))
        flight_data["dep_hour"] = h + m / 60


    # ensure duration
    duration_val = flight_data.get("elapsed")
    if duration_val is None:
        duration_val = 180  

    # Fallback defaults
    flight_data = flight_data or {
        "dest":  "JFK", "dep_hour": 12, "month": 1, "day": 1,
        "elapsed": 180, "holiday": 0, "weekday": 1,
        "temp": 20, "rain": 0, "precip": 0, "wind": 10
    }

    user_input = {
    "Carrier Code": selected_carrier or "AA",
    "Destination" : flight_data.get("dest", "JFK"),
    "DepHour"     : round(flight_data["dep_hour"], 2),
    "Month"       : flight_data.get("month", 1),
    "Day"         : flight_data.get("day",   1),
    "Duration"    : duration_val,
    "Holiday"     : flight_data.get("holiday", 0),
    "Weekday"     : flight_data.get("weekday", 1),
    "Temp"        : flight_data.get("temp", 20),
    "Rain"        : flight_data.get("rain", 0),
    "Precip"      : flight_data.get("precip", 0),
    "Wind"        : flight_data.get("wind", 10),
}

    # Model prediction
    model_row = {
    "Carrier Code"                 : user_input["Carrier Code"],
    "destination_airport"          : user_input["Destination"],
    "scheduled_dep_hour"           : user_input["DepHour"],
    "month"                        : user_input["Month"],
    "day"                          : user_input["Day"],
    "scheduled_elapsed"            : user_input["Duration"],
    "is_holiday"                   : user_input["Holiday"],
    "is_weekday"                   : user_input["Weekday"],
    "season"                       : "Winter",
    "temperature_2m_mean (°C)"     : user_input["Temp"],
    "rain_sum (mm)"                : user_input["Rain"],
    "precipitation_sum (mm)"       : user_input["Precip"],
    "wind_speed_10m_max (km/h)"    : user_input["Wind"],
}
    table_row = {
    "carrier" : user_input["Carrier Code"],
    "dest"    : user_input["Destination"],
    "dep"     : user_input["DepHour"],  
    "month"   : user_input["Month"],
    "day"     : user_input["Day"],
    "duration": user_input["Duration"],
    "holiday" : user_input["Holiday"],
    "weekday" : user_input["Weekday"],
    "temp"    : user_input["Temp"],
    "rain"    : user_input["Rain"],
    "precip"  : user_input["Precip"],
    "wind"    : user_input["Wind"],
}
    if flight_data.get("dest") is None:
        flight_data["dest"] = "JFK"

    X_user  = pd.DataFrame([model_row])

    numeric_cols = [
    'scheduled_dep_hour', 'month', 'day', 'scheduled_elapsed',
    'is_holiday', 'is_weekday',
    'temperature_2m_mean (°C)', 'rain_sum (mm)',
    'precipitation_sum (mm)', 'wind_speed_10m_max (km/h)'
    ]
    X_user[numeric_cols] = (X_user[numeric_cols]
                            .apply(pd.to_numeric, errors='coerce')
                            .fillna(0))
    
    encoded = encoder.transform(
            X_user[['Carrier Code', 'destination_airport', 'season']])
    
    X_user.fillna({"destination_airport": "JFK",
               "Carrier Code":       "AA",
               "season":             "Winter"}, inplace=True)
    
    encoded = pd.DataFrame(encoded,
                 columns=encoder.get_feature_names_out(['Carrier Code','destination_airport','season']))
    X_user_final = pd.concat(
        [X_user.drop(columns=['Carrier Code','destination_airport','season']), encoded],
        axis=1).reindex(columns=X_balanced.columns, fill_value=0)
    delay_prob = xgb_classifier.predict_proba(X_user_final)[0,1]

    # Flip‑board style table
    cols = ["carrier", "dest", "dep", "month", "day", "duration",
            "holiday", "weekday", "temp", "rain", "precip", "wind"]

    HIGHLIGHT = "#2196f3"      
    YELLOW = "#FFD600"  
    BLACK  = "#000"

    board = dash_table.DataTable(
    id      = "flip-table",
    data    = [table_row],
    editable = False,        
    columns=[
        dict(name="CARRIER",  id="carrier",  editable=False),

        dict(name="DEST",     id="dest"),

        dict(name="DEP",      id="dep",      type="numeric"),
        dict(name="MONTH",    id="month",    type="numeric"),
        dict(name="DAY",      id="day",      type="numeric"),
        dict(name="DURATION", id="duration", type="numeric"),

        dict(name="HOLIDAY",  id="holiday",  type="numeric"),
        dict(name="WEEKDAY",  id="weekday",  type="numeric"),
        dict(name="TEMP",     id="temp",     type="numeric"),
        dict(name="RAIN",     id="rain",     type="numeric"),
        dict(name="PRECIP",   id="precip",   type="numeric"),
        dict(name="WIND",     id="wind",     type="numeric"),
    ],
    style_table={"width": "100%", "overflowX": "auto"},
    style_header={
        "backgroundColor": "#000",
        "color"          : "#F7E600",
        "fontFamily"     : "PT Mono, monospace",
        "fontWeight"     : 700,
        "whiteSpace"     : "normal",   
        "textAlign"      : "center",
        "padding"        : "6px",
    },
    style_cell={
        "backgroundColor": "#000",
        "color"          : "#F7E600",
        "fontFamily"     : "PT Mono, monospace",
        "border"         : "none",
        "fontSize"       : "16px",
        "whiteSpace"     : "normal",
        "textAlign"      : "center",
        "minWidth"       : "70px",
    },
    page_action="none",
    fixed_rows={"headers": True},
    css=[                      
        {"selector": "td.dash-cell.focused, td.dash-cell.dash-cell--focused",
         "rule": f"outline:2px solid {YELLOW} !important;"
                  f"background-color:{BLACK} !important;"
                  f"color:#fff !important;"},

        {"selector": ".dash-table-container input.dash-input",
         "rule"    : f"background-color:{BLACK} !important; color:#fff !important;"},

        {"selector": ".dash-table-container select",
         "rule"    : f"background-color:{BLACK} !important; color:#fff !important;"},

        {"selector": "td.dash-cell.column-selected, td.dash-cell.dash-cell--selected",
         "rule"    : f"outline:2px solid {YELLOW} !important;"}
    ],
    style_data_conditional=[
        {"if": {"row_index": "odd"},
         "backgroundColor": "#0d0d0d"},
    ]
    )


    gauge = go.Figure(
    go.Indicator(
        mode   = "gauge+number",
        value  = delay_prob * 100,
        number = {
            "suffix": "%",
            "font": {"size": 46, "color": "white"}       
        },
        title  = {"text": "Delay Probability", "font": {"size": 18, "color": "white"}},
        domain = {"x": [0, 1], "y": [0.15, 0.80]}, 
        gauge  = {
            "shape": "angular",
            "axis": {"range": [0, 100], "visible": False},
            "steps": [
                {"range": [0, 30],  "color": "#8de37f"},
                {"range": [30, 70], "color": "#ffe867"},
                {"range": [70, 100],"color": "#f04b3a"},
            ],
            "bar": {"color": "#26418f", "thickness": 0.20},  
        },
    )
)

    gauge.update_layout(
    margin=dict(l=0, r=0, t=20, b=0),
    height=220,
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white")         
)



    report_card = dbc.Card(
        [
            dbc.CardHeader(
                "Flight Delay Prediction Report",
                className="h5",
                style={
                    "color": "#F7E600",
                    "backgroundColor": "#000",
                    "fontFamily": "PT Mono, monospace",
                },
            ),
            dbc.CardBody(board, className="p-0"),
        ],
        className="glass-card glass-card-dark-bg p-2 mb-3",
    )

    gauge_card = dbc.Card(
    dcc.Graph(figure=gauge, config={"displayModeBar": False}),
    className="glass-card glass-card-dark-bg p-2"
)

    map_fig   = build_airline_map(
        user_input["Carrier Code"], user_dest=user_input["Destination"]
    )
    map_fig.update_geos(
        showcountries=True,
        showocean=True,
        bgcolor="#000000",             
        oceancolor="#1a1a1a",
        landcolor="#414141",           
        countrycolor="#666666",
        coastlinecolor="#888888",
        showlakes=False
    )


    map_fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,                 
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            projection_type="equirectangular",
            lonaxis=dict(range=[-170, -60]),
            lataxis=dict(range=[15, 75]),
        ),
        font=dict(color="#dcdcdc")
    )

    map_graph = dbc.Card(
        dcc.Graph(
            id="airline-map",
            figure=map_fig,
            style={"height": 520},
            config={"displayModeBar": False},
        ),
        className="glass-card p-2",
    )

    content = dbc.Container(
        fluid=True,
        children=[
            
            dbc.Row(dbc.Col(report_card, width=12)),

            dbc.Row(
                [
                    dbc.Col(gauge_card, md=5, className="mb-3"),
                    dbc.Col(map_graph,  md=7),
                ]
            ),
        ],
    )


    return content

app.clientside_callback(
    """
    function(pathname) {
        var content = document.getElementById("page-content");
        if(content) {
            content.classList.remove("loaded");
            setTimeout(function() {
                content.classList.add("loaded");
            }, 100);  // You can adjust this delay
        }
        return "";
    }
    """,
    Output("dummy-output", "children"),
    Input("url", "pathname")
);
@app.callback(
    [Output("edit-mode-store", "data"),
     Output("edit-btn",        "children"),
     Output("edit-btn",        "color")],
    Input("edit-btn", "n_clicks"),
    State("edit-mode-store", "data"),
    prevent_initial_call=True
)
def toggle_edit(_, is_editing):
    is_editing = not is_editing               
    return (
        is_editing,                           
        dash.no_update,                      
        "success" if is_editing else "secondary"
    )

@app.callback(
    Output("flip-table", "editable"),
    Input("edit-mode-store", "data")
)
def set_editable(is_editing):
    return is_editing

# Run

if __name__ == '__main__':
    app.run(debug=True)
