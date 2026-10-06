from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = "uber_fare_model.joblib"
model = joblib.load(model_path)

print("Model loaded successfully.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Car Condition": request.form["car_condition"],
        "Weather": request.form["weather"],
        "Traffic Condition": request.form["traffic_condition"],
        "pickup_longitude": float(request.form["pickup_longitude"]),
        "pickup_latitude": float(request.form["pickup_latitude"]),
        "dropoff_longitude": float(request.form["dropoff_longitude"]),
        "dropoff_latitude": float(request.form["dropoff_latitude"]),
        "passenger_count": float(request.form["passenger_count"]),
        "hour": int(request.form["hour"]),
        "day": int(request.form["day"]),
        "month": int(request.form["month"]),
        "weekday": int(request.form["weekday"]),
        "year": int(request.form["year"]),
        "jfk_dist": float(request.form["jfk_dist"]),
        "ewr_dist": float(request.form["ewr_dist"]),
        "lga_dist": float(request.form["lga_dist"]),
        "sol_dist": float(request.form["sol_dist"]),
        "nyc_dist": float(request.form["nyc_dist"]),
        "distance": float(request.form["distance"]),
        "bearing": float(request.form["bearing"])
    }

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)