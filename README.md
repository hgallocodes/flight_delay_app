✈️ LAX Flight Delay Predictor






A Python-based web application that leverages machine learning and weather data to estimate the probability of flight delays for departures from Los Angeles International Airport (LAX).

DESCRIPTION

The LAX Flight Delay Predictor is a Python-based web application that leverages machine learning and weather data to estimate the probability of flight delays for departures from Los Angeles International Airport (LAX). Using historical flight records (2021–2024) combined with local weather conditions, the system trains an XGBoost classifier to model delay patterns and serves predictions through an interactive Dash dashboard.

Users interact with a guided, multi-page interface to select their airline, choose a destination airport, and specify flight details such as departure time, date, duration, and weather parameters. The app then displays the predicted delay probability with a gauge, a summary report table, and a route map, offering actionable insights for travelers.

📸 Screenshots
<p align="center"> <img src="https://github.com/user-attachments/assets/9645664a-7ab6-43d2-8957-d82bf1dbaabd" width="300" height="150" /> <img src="https://github.com/user-attachments/assets/480858af-57d6-4e30-8ac3-1fd0f724aa41" width="300" height="150" /> </p> <p align="center"> <img src="https://github.com/user-attachments/assets/7f1f63cc-9213-4e1a-8bac-04ac68f2c5a0" width="300" height="150" /> <img src="https://github.com/user-attachments/assets/980484d3-5657-4571-9be4-c35046d0638e" width="300" height="150" /> <img src="https://github.com/user-attachments/assets/b583d295-be84-45c2-93be-89a5abb248e4" width="300" height="150" /> </p>
📝 Description

The LAX Flight Delay Predictor is a data-driven dashboard that uses historical flight records (2021–2024) combined with local weather conditions to model flight delay probabilities.

Trains an XGBoost classifier to detect patterns in flight delays

Presents results through an interactive Dash web app

Helps users make informed travel decisions with visual insights

Users interact with a guided, multi-page interface to:

Select their airline

Choose a destination airport

Enter flight details (date, time, duration, weather, holiday/weekday)

View a prediction report with:

✅ Delay probability gauge

✅ Summary statistics table

✅ Route visualization map

⚙️ Installation

INSTALLATION

1. Clone the repository:

	git clone https://github.com/YOURUSERNAME/flight_delay_app

	cd flight_delay_app

3. Create a virtual environment and install dependencies:

	python3 -m venv venv

	source venv/bin/activate    # On Windows: venv\\Scripts\\activate

	pip install --upgrade pip

	pip install -r requirements.txt

5. Ensure the data/ directory contains the required CSV files:

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

🛠 Tech Stack

Frontend: Dash, Dash Bootstrap Components, Plotly

Backend: Flask

ML Model: XGBoost, Scikit-Learn

Data: FAA flight records (2021–2024), Weather data, U.S. Holidays

