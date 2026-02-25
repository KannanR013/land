from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ===============================
# LOAD TRAINED MODEL
# ===============================
data = pickle.load(open("model.pkl","rb"))

model = data["model"]
le_locality = data["le_locality"]
le_land = data["le_land"]
le_road = data["le_road"]
accuracy = round(data["accuracy"] * 100, 2)

# Auto load locality list
localities = list(le_locality.classes_)

# ===============================
# HOME PAGE
# ===============================
@app.route("/")
def home():
    return render_template(
        "index.html",
        localities=localities
    )

# ===============================
# PREDICT ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ---- GET FORM DATA ----
        area = float(request.form["area"])
        distance = float(request.form["distance"])
        locality = request.form["locality"]
        road = request.form["road"]
        land_type = request.form["land_type"]

        # ===============================
        # INDUSTRY VALIDATION
        # ===============================
        if area < 300 or area > 10000:
            return render_template(
                "result.html",
                price="Enter area between 300 and 10000 sqft",
                accuracy=accuracy
            )

        if distance < 0:
            return render_template(
                "result.html",
                price="Distance cannot be negative",
                accuracy=accuracy
            )

        # ===============================
        # ENCODE INPUTS
        # ===============================
        locality_enc = le_locality.transform([locality])[0]
        land_enc = le_land.transform([land_type])[0]
        road_enc = le_road.transform([road])[0]

        # ===============================
        # MODEL PREDICTION
        # ===============================
        prediction = model.predict(
            np.array([[area, locality_enc, distance, road_enc, land_enc]])
        )[0]

        price = f"{int(prediction):,}"

        return render_template(
            "result.html",
            price=price,
            accuracy=accuracy
        )

    except Exception as e:
        print("ERROR:", e)

        return render_template(
            "result.html",
            price="Prediction Error",
            accuracy=accuracy
        )

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run()
