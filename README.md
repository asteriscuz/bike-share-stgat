# Bike Share Demand Forecasting — ST-GAT

Spatio-temporal demand forecasting for the San Francisco Bay Area bike share network using a Graph Attention Network combined with a GRU sequence model.

## Overview

Predicting station-level bike demand is a core operational challenge for bike share systems. This project models the problem as a spatio-temporal graph forecasting task — stations are nodes, proximity defines edges, and the model learns both spatial dependencies between stations and temporal patterns over time.

Built as part of DSM 508 at IIT Indore.

## Architecture

**ST-GAT: Spatio-Temporal Graph Attention Network**

- **Graph Attention Network (GAT)** — 4 attention heads × 32 hidden dimensions, captures spatial dependencies between geographically proximate stations
- **GRU (2 layers)** — 128 hidden units, learns temporal patterns across a 12-hour sliding window
- **K-Nearest Neighbors graph** — adjacency matrix built from haversine distances between station coordinates
- **Total Parameters** — 147,225

## Features Used

- Hourly start and end demand per station
- Cyclic time encodings (hour sin/cos, day-of-week sin/cos)
- Weather features (temperature, humidity, wind speed, precipitation, cloud cover)
- Weekend/weekday flag

## Pipeline
```
Raw Trip Data → Cleaning & Outlier Removal → Feature Engineering → Graph Construction → Sliding Window Dataset → ST-GAT Training → Evaluation
```

## Results

| Metric | Score |
|--------|-------|
| R²     | 0.327 |
| MAE    | 0.87 bikes/hour |
| RMSE   | 1.48 bikes/hour |
| MSE    | 2.20 |

Model explains 32.7% of demand variance on chronologically held-out test data. Mean demand is 1.49 bikes/hour — an absolute error of 0.87 bikes provides operationally actionable forecasts for rebalancing decisions.

## Dataset

San Francisco Bay Area Bike Share Open Data (Aug 2013 – Aug 2015)

- 669,959 total trip records across 70 stations
- Filtered to 35 San Francisco weekday stations (~290,000 trips)
- Daily weather observations: temperature, humidity, wind, precipitation, cloud cover

## Tech Stack

Python · PyTorch · PyTorch Geometric · Pandas · NumPy · Scikit-learn · Folium · Matplotlib · Seaborn

## Project Structure
```
bike-share-stgat/
├── notebooks/
│   └── Bike_share_STGAT.ipynb
├── data/
│   └── README.md
└── README.md
```

## Authors

Haadiya Iman · Dipayita Basu
IIT Indore & IIM Indore
