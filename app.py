from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load("uber_fare_model.joblib")

print("Model loaded successfully.")


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance in kilometers."""
    earth_radius = 6371.0088

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    return float(
        2 * earth_radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate direction from pickup to destination in degrees."""
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    delta_lon = lon2 - lon1

    x = np.sin(delta_lon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    )

    return float((np.degrees(np.arctan2(x, y)) + 360) % 360)


def distance_to_location(lat, lon, target_lat, target_lon):
    return haversine_km(lat, lon, target_lat, target_lon)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read the selected locations from the map.
        pickup_lat = float(request.form["pickup_latitude"])
        pickup_lon = float(request.form["pickup_longitude"])
        dropoff_lat = float(request.form["dropoff_latitude"])
        dropoff_lon = float(request.form["dropoff_longitude"])

        # Read trip details.
        car_condition = request.form["car_condition"]
        weather = request.form["weather"]
        traffic_condition = request.form["traffic_condition"]
        passenger_count = int(request.form["passenger_count"])

        # Extract date and time features.
        trip_datetime = datetime.fromisoformat(
            request.form["pickup_datetime"]
        )

        # Calculate trip distance and bearing.
        distance = haversine_km(
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
        )

        bearing = calculate_bearing(
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
        )

        # Convert coordinates from degrees to radians,
        # matching the coordinate format used in the training data.
        pickup_lat_rad = np.radians(pickup_lat)
        pickup_lon_rad = np.radians(pickup_lon)
        dropoff_lat_rad = np.radians(dropoff_lat)
        dropoff_lon_rad = np.radians(dropoff_lon)

        # Reference locations in New York.
        # These distances are estimates; replace them with the
        # exact Task 1 calculations if those become available.
        jfk_dist = distance_to_location(
            pickup_lat, pickup_lon, 40.6413, -73.7781
        )
        ewr_dist = distance_to_location(
            pickup_lat, pickup_lon, 40.6895, -74.1745
        )
        lga_dist = distance_to_location(
            pickup_lat, pickup_lon, 40.7769, -73.8740
        )
        sol_dist = distance_to_location(
            pickup_lat, pickup_lon, 40.6892, -74.0445
        )
        nyc_dist = distance_to_location(
            pickup_lat, pickup_lon, 40.7128, -74.0060
        )

        # Prepare all model features.
        data = {
            "Car Condition": car_condition,
            "Weather": weather,
            "Traffic Condition": traffic_condition,
            "pickup_longitude": pickup_lon_rad,
            "pickup_latitude": pickup_lat_rad,
            "dropoff_longitude": dropoff_lon_rad,
            "dropoff_latitude": dropoff_lat_rad,
            "passenger_count": passenger_count,
            "hour": trip_datetime.hour,
            "day": trip_datetime.day,
            "month": trip_datetime.month,
            "weekday": trip_datetime.weekday(),
            "year": trip_datetime.year,
            "jfk_dist": jfk_dist,
            "ewr_dist": ewr_dist,
            "lga_dist": lga_dist,
            "sol_dist": sol_dist,
            "nyc_dist": nyc_dist,
            "distance": distance,
            "bearing": bearing
        }

        input_data = pd.DataFrame([data])

        # Predict using the already-trained pipeline.
        prediction = float(model.predict(input_data)[0])

        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error=f"Could not predict the fare. Please check your inputs. Details: {e}"
        ), 400


if __name__ == "__main__":
    app.run(debug=True)

