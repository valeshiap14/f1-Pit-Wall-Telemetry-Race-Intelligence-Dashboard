# F1 Pit Wall - Telemetry & Race Intelligence Dashboard

## About the Project

F1 Pit Wall is a Formula 1 telemetry and race analysis project I built using Python and real Formula 1 data.

The idea behind this project was to build something that goes further than just displaying lap times and basic race statistics. I wanted to understand what is actually happening around a lap and use telemetry to answer questions such as where one driver is faster, who brakes later, who carries more speed through a corner and where lap time is being gained or lost.

The dashboard allows Formula 1 sessions to be loaded and analysed using real data from FastF1.

I also included tyre degradation analysis, track evolution, race strategy analysis, automated race engineer insights and machine learning for lap-time prediction.

The overall flow of the project is:

```text
Formula 1 Data
       ↓
Data Cleaning
       ↓
Data Processing
       ↓
Feature Engineering
       ↓
Telemetry Analysis
       ↓
Statistical Analysis
       ↓
Machine Learning
       ↓
Data Visualisation
       ↓
Race Intelligence
```

---

## Why I Built This

I wanted to build a Data Science project around something I am actually interested in, so Formula 1 gave me a good opportunity to work with a large real-world dataset while learning more about telemetry and motorsport analytics.

Instead of creating a dashboard that only displays charts, I wanted the project to actually analyse the data.

Some of the questions I wanted the dashboard to answer were:

- Where is one driver faster than another?
- Where is lap time being gained or lost?
- Who brakes later?
- Who carries more speed into a corner?
- Who has the higher minimum corner speed?
- Who gets back onto the throttle earlier?
- Who has better straight-line speed?
- How quickly are the tyres degrading?
- How does performance change throughout a stint?
- How does the track change during a session?
- How do different race strategies compare?
- Can machine learning estimate a future lap time?

The goal was basically to build my own smaller version of a motorsport performance engineering and race intelligence platform.

---

# Technologies Used

The main technologies used in this project are:

- Python
- FastF1
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-learn
- SciPy
- SQLite
- Pytest
- GitHub

---

# Dashboard

The dashboard is built using Streamlit.

I designed it to feel more like an engineering or pit-wall interface instead of just using a basic dashboard layout.

The application is split into different analysis sections so I can look at the same Formula 1 session from different perspectives.

The main areas of the project include:

1. Race Overview
2. Driver Comparison
3. Telemetry
4. Track Map
5. Corner Analysis
6. Tyre Performance
7. Track Evolution
8. Strategy Analysis
9. Predictive Analytics
10. Session Insights
11. Data Quality

---

# Race Overview

The Race Overview gives a quick summary of the loaded Formula 1 session.

It can display information such as:

- Season
- Grand Prix
- Circuit
- Session
- Selected drivers
- Fastest lap
- Lap gap
- Pace advantage
- Track temperature
- Air temperature
- Humidity
- Weather conditions

This gives some context before going deeper into the telemetry.

---

# Driver Comparison

Two drivers can be selected and compared using calculated performance metrics.

The comparison can include:

- Fastest lap
- Sector times
- Lap number
- Tyre compound
- Maximum speed
- Average speed
- Full-throttle percentage
- Number of braking events
- Average RPM

This gives me a quick picture of the differences between the drivers before looking at the full telemetry.

---

# Telemetry Analysis

The telemetry section is one of the main parts of the project.

Telemetry from both drivers is aligned using lap distance so that their values can be compared at approximately the same physical location around the circuit.

The dashboard can compare:

### Speed

Speed in km/h throughout the lap.

### Throttle

How much throttle each driver is applying.

### Brake

Where each driver is braking.

### Gear

The gear being used at different points around the lap.

### RPM

Engine RPM throughout the lap.

### DRS

DRS information where it is available in the telemetry.

### Lap Delta

The estimated time difference between the two drivers throughout the lap.

Using the same distance axis makes it easier to compare different telemetry channels and understand what both drivers are doing at the same part of the circuit.

---

# Lap Delta Analysis

One thing I specifically wanted to calculate was lap delta.

Knowing that one driver finished a lap 0.2 seconds faster does not tell us where that difference came from.

The lap delta helps show where the advantage develops around the circuit.

For example, one driver might lose time under braking but gain it back by carrying more speed through the corner or getting onto the throttle earlier.

This makes the comparison much more useful than only comparing final lap times.

---

# Track Map

The project uses position telemetry to reconstruct the circuit.

The track can then be visualised using different telemetry values such as:

- Speed
- Throttle
- Gear
- RPM
- Lap performance information where available

This helps connect the normal telemetry graphs to actual areas around the circuit.

Instead of only seeing that a speed difference happened at a certain distance, I can also understand approximately where on the track it happened.

---

# Corner Analysis

I created a corner analysis system that attempts to identify corner regions using telemetry.

For each detected corner region, the application can calculate values such as:

- Entry speed
- Minimum corner speed
- Exit speed
- Approximate braking point
- Throttle application
- Approximate full-throttle point
- Gear
- Approximate time gained or lost

The application can then generate a short explanation based on the calculated values.

For example:

> Driver A gained time through the analysed corner by braking later, maintaining a higher minimum speed and returning to full throttle earlier.

I deliberately treat these measurements as estimates because telemetry resolution does not always support extremely precise conclusions.

The detected corner regions also do not necessarily match the official circuit corner boundaries.

---

# Tyre Performance

The tyre section analyses individual tyre stints.

Before calculating degradation, the cleaning process attempts to identify laps that may not represent normal pace.

These can include:

- Pit laps
- In laps
- Out laps
- Safety Car affected laps
- Invalid laps
- Statistical outliers
- Weather affected laps where detectable

The tyre analysis can calculate:

- Tyre compound
- Stint
- Stint length
- Number of clean laps
- Best lap
- Average lap
- Estimated tyre degradation
- Seconds lost per lap
- Model error

Regression models can then be used to estimate how lap performance changes as tyre age increases.

I also wanted to avoid automatically assuming that every tyre stint degrades in a perfectly straight line, because real tyre behaviour can be affected by many different factors.

---

# Track Evolution

Track conditions can change throughout a Formula 1 session.

The Track Evolution section looks at how lap performance changes during the session and compares it with available environmental information.

This can include:

- Session time
- Lap time
- Track temperature
- Air temperature
- Rainfall
- Tyre information

I tried to be careful about correlation and causation in this section.

For example, if lap times become faster while track temperature increases, that does not automatically mean the temperature caused the lap-time improvement.

Other factors such as fuel load, tyres, traffic, track grip, driver improvement and team run plans could also affect the result.

---

# Strategy Analysis

The Strategy Analysis section looks at race stint information.

It can compare:

- Stint lengths
- Tyre compounds
- Pit-stop timing
- Pace before stops
- Pace after stops
- Driver stint timelines
- Degradation during the stint

The dashboard can display the tyre strategies of selected drivers and compare their pace throughout the race.

This is currently a simplified strategy analysis and is not intended to reproduce the full strategy simulation systems used by Formula 1 teams.

Any hypothetical strategy conclusions should therefore be treated as estimates.

---

# Predictive Analytics

I also added machine learning to the project.

The goal is to experiment with predicting lap performance using information available from the session.

The models currently include:

- Linear Regression
- Ridge Regression
- Random Forest Regression

Instead of training one model and automatically assuming it is good, I compare the models using evaluation metrics.

These include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² where appropriate

The model results can then be compared before using a model to generate an estimated lap time.

---

## Preventing Data Leakage

One thing I had to consider when building the machine learning section was data leakage.

Formula 1 session data happens in chronological order.

Randomly mixing earlier and later laps could allow information from the future to influence predictions made on earlier observations.

To reduce this problem, the data can be split chronologically.

Earlier observations are used for training while later observations are kept for evaluation.

This gives a more realistic test of how well the model performs.

---

# Automated Race Engineer Insights

Another feature I added was an automated race engineer insight system.

The idea is to take values already calculated by the application and convert them into easier-to-read observations.

The system can generate insights about:

- Lap-time differences
- Straight-line speed
- Braking
- Throttle
- Corner performance
- Tyre degradation
- Track evolution
- Strategy
- Machine-learning performance

An important rule I followed here is that the insight system should only describe values calculated by the application.

It should not invent telemetry values that are not available in the data.

---

# Data Quality

Real motorsport data is not always perfectly clean.

Because of this, I created a separate data-quality and cleaning pipeline.

The application can identify or handle things such as:

- Missing values
- Duplicate observations
- Invalid laps
- Incomplete laps
- Pit laps
- Outliers
- Safety Car affected laps
- Weather interruptions
- Abnormal lap times

Instead of simply deleting questionable laps, I also use quality flags.

This means I can understand why a lap was excluded from a calculation.

I found this useful because it makes the analysis more traceable and easier to debug.

---

# Formula 1 Data

The Formula 1 data is loaded using FastF1.

Depending on the session and season, the available data can include:

- Lap times
- Sector times
- Speed
- Throttle
- Brake
- RPM
- Gear
- DRS
- Position data
- Tyre compounds
- Tyre age
- Stints
- Pit information
- Track status
- Weather
- Session timing

The amount and quality of telemetry available can vary depending on the session.

FastF1 also supports local caching, which is important for this project because Formula 1 telemetry sessions can contain a large amount of data.

---

# Data Storage and GitHub File Size

The actual Formula 1 telemetry used by this project is not stored directly inside this GitHub repository.

While developing and testing the project, FastF1 downloaded timing, telemetry, position and session information for the Formula 1 sessions I was analysing.

After testing multiple sessions, the local project data grew to over 100 MB.

Most of this size came from the FastF1 cache rather than my actual Python source code.

Because the FastF1 data can be downloaded again automatically, I decided there was no reason to upload all of the cached telemetry to GitHub.

The repository therefore contains the code needed to retrieve, process and analyse the Formula 1 data instead of containing every downloaded telemetry file.

When the dashboard is used locally, FastF1 stores downloaded session data inside:

```text
data/cache/
```

This means the GitHub repository can stay lightweight while the application can still work with real Formula 1 data.

---

## Files Not Included in the Repository

Some files and folders that exist on my development machine are intentionally not included in the GitHub version of the project.

These include:

```text
data/cache/
data/f1_pitwall.db
.venv/
.idea/
__pycache__/
.pytest_cache/
```

These files are either generated automatically or only required for my local development environment.

### FastF1 Cache

```text
data/cache/
```

This contains downloaded Formula 1 timing and telemetry data.

The cache can become quite large, especially after analysing multiple races.

FastF1 can recreate this data when a session is loaded again.

### SQLite Database

```text
data/f1_pitwall.db
```

The application can create a local SQLite database for processed analytical information.

The database is generated locally, so I do not need to store the generated database file in the repository.

### Virtual Environment

```text
.venv/
```

The Python virtual environment can also become quite large because it contains installed Python packages.

Instead of uploading those packages, the dependencies are listed inside:

```text
requirements.txt
```

They can be installed again using:

```bash
python -m pip install -r requirements.txt
```

### PyCharm Files

```text
.idea/
```

These are local PyCharm project settings and are not required to run the actual application.

### Python Cache Files

Folders such as:

```text
__pycache__/
.pytest_cache/
```

are automatically generated by Python and Pytest.

They are also not required in the GitHub repository.

---

# Reproducing the Data Locally

The large FastF1 cache is not required to clone the project.

After installing the dependencies, run:

```bash
python -m streamlit run app.py
```

Then select:

1. Season
2. Grand Prix
3. Session
4. Load the session

FastF1 will retrieve the required Formula 1 information.

The first time a session is loaded may take longer because the data needs to be downloaded and processed.

Future loads of the same session should normally be faster because the local cache can be reused.

This approach keeps large generated datasets out of GitHub while keeping the project reproducible.

---

# Drivers Used During Testing

One of the main sessions I used while developing and testing the dashboard was the **2025 Monaco Grand Prix Qualifying session**.

The session included the following drivers:

| Driver | Code | Team |
|---|---|---|
| Lando Norris | NOR | McLaren |
| Charles Leclerc | LEC | Ferrari |
| Oscar Piastri | PIA | McLaren |
| Lewis Hamilton | HAM | Ferrari |
| Max Verstappen | VER | Red Bull Racing |
| Isack Hadjar | HAD | Racing Bulls |
| Fernando Alonso | ALO | Aston Martin |
| Esteban Ocon | OCO | Haas |
| Liam Lawson | LAW | Racing Bulls |
| Alexander Albon | ALB | Williams |
| Carlos Sainz | SAI | Williams |
| Yuki Tsunoda | TSU | Red Bull Racing |
| Nico Hülkenberg | HUL | Sauber |
| George Russell | RUS | Mercedes |
| Kimi Antonelli | ANT | Mercedes |
| Gabriel Bortoleto | BOR | Sauber |
| Oliver Bearman | BEA | Haas |
| Pierre Gasly | GAS | Alpine |
| Lance Stroll | STR | Aston Martin |
| Franco Colapinto | COL | Alpine |

These drivers are not hard-coded into the dashboard.

The available driver list comes from the Formula 1 session that has been loaded, so it can change depending on the season, Grand Prix and session selected.

---

# SQLite Database

I added SQLite so processed analytical results can be stored locally.

The database can be used to store information such as:

- Session information
- Analytical results
- Tyre analysis
- Model information
- Generated race engineer insights

The generated database file is kept locally and is not included in the GitHub repository.

I structured the project so that something like PostgreSQL could also be added later if the application grows.

---

# Project Structure

I did not want to put the entire project inside one large `app.py` file.

Instead, I separated different parts of the analysis into their own modules.

A simplified version of the project structure looks like this:

```text
f1-pitwall/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│
├── data/
│   ├── cache/
│   └── processed/
│
├── models/
│
├── src/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── corners.py
│   ├── data_loader.py
│   ├── database.py
│   ├── insights.py
│   ├── predictions.py
│   ├── session_service.py
│   ├── strategy.py
│   ├── telemetry.py
│   ├── telemetry_charts.py
│   ├── track_evolution.py
│   ├── track_evolution_charts.py
│   ├── track_map.py
│   ├── tyres.py
│   ├── tyre_charts.py
│   ├── utils.py
│   └── weather.py
│
└── tests/
    ├── test_cleaning.py
    ├── test_corners.py
    ├── test_lap_delta.py
    ├── test_predictions.py
    ├── test_strategy.py
    ├── test_track_evolution.py
    └── test_tyres.py
```

The basic idea behind the architecture is:

```text
Streamlit Dashboard
        ↓
Session Service
        ↓
Analytics Modules
        ↓
Cleaning / Telemetry / Strategy / ML
        ↓
FastF1 + SQLite
```

I structured it this way so the analytical code is not completely dependent on Streamlit.

This should also make it easier to introduce something like FastAPI or PostgreSQL later without rebuilding the whole project.

---

# Installation

Clone or download the repository.

If using Git:

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project folder:

```bash
cd f1-pitwall
```

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows, activate it using:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

You do not need to manually download the Formula 1 telemetry included in my local development cache.

FastF1 will retrieve the required session information when the dashboard loads a session for the first time.

---

# Running the Dashboard

Start the Streamlit application with:

```bash
python -m streamlit run app.py
```

Streamlit should open the dashboard in the browser.

From there:

1. Select a season.
2. Select a Grand Prix.
3. Select a session.
4. Load the session.
5. Select the drivers.
6. Choose the analysis you want to explore.

The first session load may take some time because FastF1 needs to retrieve and process the data.

---

# Testing

I created automated tests for some of the most important analytical calculations in the project.

The tests can be run using:

```bash
python -m pytest tests -v
```

Testing covers areas such as:

- Telemetry alignment
- Lap delta
- Corner detection
- Corner analysis
- Tyre degradation
- Outlier removal
- Data-quality flags
- Strategy calculations
- Track-evolution calculations
- Machine-learning preprocessing

During development I used these tests to make sure changes to one part of the project did not silently break important analytical calculations.

---

# Screenshots

Screenshots of the dashboard can be added here.

## Race Overview

`Screenshot coming soon`

## Driver Comparison

`Screenshot coming soon`

## Telemetry

`Screenshot coming soon`

## Track Map

`Screenshot coming soon`

## Corner Analysis

`Screenshot coming soon`

## Tyre Performance

`Screenshot coming soon`

## Track Evolution

`Screenshot coming soon`

## Strategy Analysis

`Screenshot coming soon`

## Predictive Analytics

`Screenshot coming soon`

---

# Challenges

Some of the more difficult parts of building this project were:

- Working with large telemetry datasets
- Aligning telemetry from two different drivers
- Calculating lap delta
- Handling missing FastF1 telemetry
- Detecting corners from telemetry
- Identifying representative tyre laps
- Removing outliers without removing useful data
- Analysing track evolution
- Separating the analytics code from the dashboard
- Preventing machine-learning data leakage
- Turning calculated values into readable race engineer insights
- Managing the size of downloaded Formula 1 telemetry

These were also some of the areas where I learned the most while working on the project.

---

# Limitations

This project is not intended to reproduce the actual software or simulation systems used by Formula 1 teams.

There are several limitations that need to be considered.

Telemetry availability and resolution can vary between sessions.

Detected corner regions may not exactly match official circuit corner boundaries.

Braking points and throttle points are estimates based on the available telemetry resolution.

Tyre degradation can also be affected by many factors including:

- Fuel load
- Traffic
- Driver management
- Track evolution
- Weather
- Safety Cars
- Tyre warm-up

Because of this, tyre degradation values should be treated as analytical estimates rather than perfect measurements.

The machine-learning models are experimental statistical models based on available data and should not be treated as professional Formula 1 simulation models.

The strategy analysis also does not currently model every variable involved in a real Formula 1 race.

---

# Future Improvements

There are still a lot of things I would like to experiment with and add to the project.

Some future ideas include:

- FastAPI backend
- PostgreSQL database
- Docker
- Cloud deployment
- Better corner detection
- Official corner information
- Multi-lap telemetry comparison
- Racing-line comparison
- Better braking analysis
- Fuel-corrected pace
- Traffic detection
- Tyre warm-up modelling
- Undercut analysis
- Overcut analysis
- Safety Car strategy simulation
- Pit-loss modelling
- Monte Carlo race simulations
- Feature importance
- SHAP model explanations
- Models trained across multiple races
- Circuit-specific machine-learning models
- Automated race reports
- GitHub Actions for automated testing

---

# What I Learned

This project gave me experience working on more than just an isolated machine-learning model or a normal dashboard.

I had to think about the entire process:

```text
Getting the data
      ↓
Cleaning the data
      ↓
Processing telemetry
      ↓
Engineering useful features
      ↓
Analysing performance
      ↓
Testing calculations
      ↓
Building ML models
      ↓
Evaluating the models
      ↓
Visualising the results
      ↓
Building an application around everything
```

One of the biggest things I learned while working on this project is that creating a graph or prediction is only one part of Data Science.

The result also needs to be understandable, testable and connected back to the original data.

I also learned how important it is to avoid making conclusions that are more precise than the data actually supports.

---

# Main Skills Demonstrated

## Data Science

- Data cleaning
- Data analysis
- Exploratory analysis
- Feature engineering
- Regression
- Statistical analysis
- Model evaluation

## Machine Learning

- Linear Regression
- Ridge Regression
- Random Forest
- Training and testing
- Chronological validation
- MAE
- RMSE
- Prediction pipelines

## Data Engineering

- Loading external data
- Processing telemetry
- Local caching
- Data transformation
- SQLite storage
- Reusable data pipelines

## Data Visualisation

- Plotly
- Interactive telemetry charts
- Circuit maps
- Lap delta visualisation
- Tyre degradation charts
- Strategy timelines

## Software Engineering

- Modular Python
- Reusable functions
- Error handling
- Automated testing
- Project structure
- GitHub

## Motorsport Analytics

- Formula 1 telemetry
- Driver comparison
- Corner analysis
- Braking analysis
- Throttle analysis
- Tyre degradation
- Track evolution
- Race strategy

---

# Final Goal

The main goal of F1 Pit Wall is not just to display Formula 1 charts.

I wanted to show how raw motorsport data can be taken through an entire Data Science process and eventually turned into useful performance information.

```text
Raw Formula 1 Data
        ↓
Clean Data
        ↓
Telemetry
        ↓
Performance Metrics
        ↓
Statistical Analysis
        ↓
Machine Learning
        ↓
Interactive Visualisations
        ↓
Race Intelligence
```

The project is something I can continue improving as I learn more about Data Science, machine learning, software engineering and motorsport analytics.

---

# Disclaimer

F1 Pit Wall is an independent educational and portfolio project.

It is not affiliated with, endorsed by or associated with Formula 1, the FIA, any Formula 1 team, driver or FastF1.

Formula 1 related names, team names, driver names, event names and trademarks belong to their respective owners.