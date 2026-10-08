# Uber Fare Prediction — Model Deployment

A Flask-based web application for deploying a machine learning model that predicts Uber trip fares based on trip, location, time, and distance-related features.

## Project Overview

This project represents **Task 3 — Model Deployment** of the Uber Fare Prediction project.

The best-performing model from Task 2, a **Tuned Random Forest Regressor**, was saved as a Scikit-learn pipeline and integrated into a Flask web application.

The application allows users to enter trip information through a web interface and receive a predicted fare.

## Model

The deployed model is a **Tuned Random Forest Regressor**.

### Hyperparameters

* `n_estimators = 200`
* `max_depth = 30`
* `min_samples_split = 5`
* `min_samples_leaf = 2`
* `max_features = sqrt`

### Performance

| Metric |  Score |
| ------ | -----: |
| MAE    | 1.6465 |
| RMSE   | 3.4732 |
| R²     | 0.8671 |

The model achieved the best performance compared with Linear Regression and Decision Tree Regressor during Task 2 evaluation.

## Features

The model uses 20 input features:

* Car Condition
* Weather
* Traffic Condition
* Pickup Longitude
* Pickup Latitude
* Dropoff Longitude
* Dropoff Latitude
* Passenger Count
* Hour
* Day
* Month
* Weekday
* Year
* JFK Distance
* EWR Distance
* LGA Distance
* SOL Distance
* NYC Distance
* Trip Distance
* Bearing

## Technologies

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* Joblib
* HTML
* CSS
* Git LFS

## Project Structure

```text
uber-fare-prediction-deployment/
│
├── app.py
├── requirements.txt
├── uber_fare_model.joblib
├── task_3_test.ipynb
├── templates/
│   └── index.html
└── README.md
```

## How It Works

```text
User Input
    ↓
Flask Web Interface
    ↓
Pandas DataFrame
    ↓
Saved Scikit-learn Pipeline
    ↓
Preprocessing
    ↓
Tuned Random Forest
    ↓
Predicted Fare
```

The trained model is loaded once when the Flask application starts. It does not retrain the model for every prediction.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/mo7amed3ali/uber-fare-prediction-deployment.git
cd uber-fare-prediction-deployment
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Deployment

The application is designed to be deployed as a Flask Web Service using a production WSGI server such as Gunicorn.

Start command:

```bash
gunicorn app:app
```

## Project Goal

The goal of this project is to demonstrate the complete transition from a trained machine learning model to a usable web application, including model serialization, Flask integration, user input handling, prediction, and deployment preparation.

## Author

**Mohamed Ali**

Computer Science Student | Data Engineer | AI & Machine Learning Enthusiast

GitHub: [mo7amed3ali](https://github.com/mo7amed3ali)
