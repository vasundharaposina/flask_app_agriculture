import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import requests
import json

app = Flask(__name__)

# ---------------- MODEL ----------------
model = tf.keras.models.load_model("leaf_disease_model.h5")

# 🔴 MUST MATCH MODEL TRAINING ORDER
CLASS_NAMES = [
    "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites",
    "Tomato_Target_spot", "Tomato_Yellow_leaf_curl_virus",
    "Tomato_Mosaic_virus",
    "Potato_Early_blight", "Potato_Late_blight"
]

# ---------------- LANGUAGE FILE ----------------
LANG = json.load(open("language.json", encoding="utf-8"))

# ---------------- WEATHER ----------------
WEATHER_API_KEY = "d87f6ee5e35a1ffc55ea151a5d55c64c"
CITY = "Bangalore"

# ---------------- GRAPH DATA (GLOBAL) ----------------
DAYS = ["Day 1", "Day 2", "Day 3", "Day 4", "Today"]
TEMPS = [29, 30, 28, 31, 30]
HUMIDITIES = [65, 70, 60, 75, 68]
RAINFALLS = [2, 5, 0, 3, 1]

# ---------------- DASHBOARD ----------------
@app.route("/")
def dashboard():
    image = request.args.get("image")
    prediction = request.args.get("prediction")
    solution = request.args.get("solution")

    # Weather API
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={WEATHER_API_KEY}&units=metric"
        data = requests.get(url).json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)
    except:
        temp, humidity, rainfall = 30, 70, 0

    return render_template(
        "dashboard.html",
        city=CITY,
        temp=temp,
        humidity=humidity,
        rainfall=rainfall,

        days=DAYS,
        temps=TEMPS,
        humidities=HUMIDITIES,
        rainfalls=RAINFALLS,

        image=image,
        prediction=prediction,
        solution=solution
    )

# ---------------- MANUAL IMAGE UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("leaf")
    lang = request.form.get("lang", "en")

    if not file:
        return redirect(url_for("dashboard"))

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    path = os.path.join(upload_dir, filename)
    file.save(path)

    # Preprocess
    img = Image.open(path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    prediction = CLASS_NAMES[idx]

    crop = prediction.split("_")[0].capitalize()
    disease = "_".join(prediction.split("_")[1:]).replace("_", " ").title()

    solution = LANG.get(crop, {}).get(disease, {}).get(lang, "Solution not available")

    return render_template(
        "dashboard.html",
        city=CITY,
        temp=TEMPS[-1],
        humidity=HUMIDITIES[-1],
        rainfall=RAINFALLS[-1],

        days=DAYS,
        temps=TEMPS,
        humidities=HUMIDITIES,
        rainfalls=RAINFALLS,

        image="uploads/" + filename,
        prediction=prediction,
        solution=solution
    )

# ---------------- ESP32 CAM IMAGE UPLOAD ----------------
@app.route("/esp32_upload", methods=["POST"])
def esp32_upload():
    if "image" not in request.files:
        return jsonify({"success": False, "msg": "No image received"})

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    image_file = request.files["image"]
    image_path = os.path.join(upload_dir, "esp32_leaf.jpg")
    image_file.save(image_path)

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    prediction = CLASS_NAMES[idx]

    crop = prediction.split("_")[0].capitalize()
    disease = "_".join(prediction.split("_")[1:]).replace("_", " ").title()

    solution = LANG.get(crop, {}).get(disease, {}).get("en", "Solution not available")

    return jsonify({
        "success": True,
        "image": "uploads/esp32_leaf.jpg",
        "prediction": prediction,
        "solution": solution
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)