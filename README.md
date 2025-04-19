DESCRIPTION

The LAX Flight Delay Predictor is a Python-based web application that leverages machine learning and weather data to estimate the probability of flight delays for departures from Los Angeles International Airport (LAX). Using historical flight records (2021–2024) combined with local weather conditions, the system trains an XGBoost classifier to model delay patterns and serves predictions through an interactive Dash dashboard.

Users interact with a guided, multi-page interface to select their airline, choose a destination airport, and specify flight details such as departure time, date, duration, and weather parameters. The app then displays the predicted delay probability with a gauge, a summary report table, and a route map, offering actionable insights for travelers.

INSTALLATION

1. Clone the repository:

	git clone https://github.com/YOURUSERNAME/flight_delay_app
	cd flight_delay_app

2. Create a virtual environment and install dependencies:

	python3 -m venv venv
	source venv/bin/activate    # On Windows: venv\\Scripts\\activate
	pip install --upgrade pip
	pip install -r requirements.txt
3. Ensure the data/ directory contains the required CSV files:

	- Airline datasets: AA.csv, AS.csv, DL.csv, UA.csv, WN.csv, SA.csv, JB.csv
	- LA weather: la_weather.csv (with header rows trimmed)
	- Global airport database: GlobalAirportDatabase.txt

EXECUTION

1. Activate your virtual environment (if not already active):
	
	source venv/bin/activate    # On Windows: venv\\Scripts\\activate

2. Run the application:
	
	python app.py

3. Open a web browser and navigate to http://127.0.0.1:8050/.

4. Follow the on-screen steps:

	- Select an airline using the logo buttons.

	- Choose your destination airport.

	- Enter flight details: date, departure time, duration, holiday/weekday, and weather 		  sliders.

	- View the delay probability, data summary, and route map on the final page.

Enjoy exploring and planning your flights with data-driven insights!



