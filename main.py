from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
import joblib
import warnings
import uvicorn
import indiapins
from flask import jsonify
from flask import request
import requests
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

app = FastAPI()

# Define file paths for model and scaler
MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"
DATA_PATH = "Crop_recommendation.csv"

# --- Model Training and Saving (Run only if model/scaler don't exist) ---
# This block ensures the model is trained and saved only once,
# preventing retraining on every application startup.
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Model or scaler not found. Training and saving the model...")
    try:
        df = pd.read_csv(DATA_PATH)

        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = df[features]
        y = df['label']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, MODEL_PATH)
        print(f"Trained model saved as {MODEL_PATH}")

        # Save the scaler
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved as {SCALER_PATH}")

        # Display unique crop labels to check for imbalance
        print("Unique Crops and their counts:\n", y.value_counts())

    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Please ensure the dataset is in the correct directory.")
        model = None
        scaler = None
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        model = None
        scaler = None
else:
    print("Model and scaler already exist. Skipping training.")


# --- Global Model and Scaler Loading ---
# Load the trained model and scaler once when the application starts
# This makes them available globally for prediction endpoints.
model = None
scaler = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}.")
    else:
        print(f"⚠️ {MODEL_PATH} not found. Crop prediction feature will not work.")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded successfully from {SCALER_PATH}.")
    else:
        print(f"⚠️ {SCALER_PATH} not found. Crop prediction feature will not work.")

except Exception as e:
    print(f"❌ Error loading model or scaler: {e}. Crop prediction feature will not work.")


# --- Static Files and Templates Setup ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Routes for Navigation ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
async def read_rt(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/recommendations", response_class=HTMLResponse)
async def recommendation(request: Request):
    return templates.TemplateResponse("recommendations.html", {"request": request})

@app.get("/soil-analysis", response_class=HTMLResponse)
async def soil_analysis_form(request: Request):
    """Displays the soil analysis input form."""
    return templates.TemplateResponse("soil-analysis.html", {"request": request})

@app.get("/fertilizer", response_class=HTMLResponse)
async def fertilizer(request: Request):
    return templates.TemplateResponse("fertilizer.html", {"request": request})

@app.get("/weather", response_class=HTMLResponse)
async def weather(request: Request):
    return templates.TemplateResponse("weather.html", {"request": request})

@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations(request: Request):
    return templates.TemplateResponse("recommendations.html", {"request": request})

@app.post("/submit-form", response_class=HTMLResponse)
async def submit_form(request: Request, name: str = Form(...), email: str = Form(...)):
    return templates.TemplateResponse("form-submitted.html", {"request": request, "name": name, "email": email})


# --- Soil Analysis Logic ---
@app.post("/analyze-soil", response_class=HTMLResponse)
async def analyze_soil(
    request: Request,
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    ph: float = Form(...)
):
    """
    Analyzes soil nutrient levels and pH, providing recommendations.
    """
    optimal_ranges = {
        "nitrogen": {"min": 40, "max": 80},
        "phosphorus": {"min": 20, "max": 50},
        "potassium": {"min": 100, "max": 300},
        "ph": {"min": 6.0, "max": 7.5}
    }

    analysis_results = {
        "nitrogen_status": "Optimal" if optimal_ranges["nitrogen"]["min"] <= nitrogen <= optimal_ranges["nitrogen"]["max"] else "Needs Attention",
        "phosphorus_status": "Optimal" if optimal_ranges["phosphorus"]["min"] <= phosphorus <= optimal_ranges["phosphorus"]["max"] else "Needs Attention",
        "potassium_status": "Optimal" if optimal_ranges["potassium"]["min"] <= potassium <= optimal_ranges["potassium"]["max"] else "Needs Attention",
        "ph_status": "Optimal" if optimal_ranges["ph"]["min"] <= ph <= optimal_ranges["ph"]["max"] else "Needs Attention",
        "ph_recommendation": "",
        "nitrogen_recommendation": "",
        "phosphorus_recommendation": "",
        "potassium_recommendation": ""
    }

    soil_health_score = 0
    if analysis_results["nitrogen_status"] == "Optimal":
        soil_health_score += 1
    if analysis_results["phosphorus_status"] == "Optimal":
        soil_health_score += 1
    if analysis_results["potassium_status"] == "Optimal":
        soil_health_score += 1
    if analysis_results["ph_status"] == "Optimal":
        soil_health_score += 1

    if soil_health_score == 4:
        soil_health = "Excellent"
    elif soil_health_score >= 2:
        soil_health = "Good"
    else:
        soil_health = "Poor"

    if analysis_results["ph_status"] == "Needs Attention":
        if ph < optimal_ranges["ph"]["min"]:
            analysis_results["ph_recommendation"] = "Soil is too acidic. Consider adding lime to increase pH."
        else:
            analysis_results["ph_recommendation"] = "Soil is too alkaline. Consider adding organic matter or sulfur to decrease pH."

    if analysis_results["nitrogen_status"] == "Needs Attention":
        if nitrogen < optimal_ranges["nitrogen"]["min"]:
            analysis_results["nitrogen_recommendation"] = "Nitrogen is low. Consider adding nitrogen-rich fertilizers (e.g., urea, composted manure)."
        else:
            analysis_results["nitrogen_recommendation"] = "Nitrogen is high. Avoid nitrogen fertilizers for a while; excess can harm plants."

    if analysis_results["phosphorus_status"] == "Needs Attention":
        if phosphorus < optimal_ranges["phosphorus"]["min"]:
            analysis_results["phosphorus_recommendation"] = "Phosphorus is low. Use phosphorus-rich fertilizers like bone meal or rock phosphate."
        else:
            analysis_results["phosphorus_recommendation"] = "Phosphorus is high. Be cautious with phosphorus fertilizers; excess can lead to nutrient imbalances."

    if analysis_results["potassium_status"] == "Needs Attention":
        if potassium < optimal_ranges["potassium"]["min"]:
            analysis_results["potassium_recommendation"] = "Potassium is low. Apply potassium-rich fertilizers (e.g., potash, wood ash)."
        else:
            analysis_results["potassium_recommendation"] = "Potassium is high. Generally less problematic, but monitor other nutrient levels."

    return templates.TemplateResponse("soil-analysis.html", {
        "request": request,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "ph": ph,
        "soil_health": soil_health,
        "analysis_results": analysis_results
    })


# --- Crop Prediction Logic ---
@app.get("/crop-prediction", response_class=HTMLResponse)
async def get_crop_prediction_form(request: Request):
    """Displays the form for crop prediction."""
    return templates.TemplateResponse("crop-prediction-form.html", {"request": request})

@app.post("/predict-crop", response_class=HTMLResponse)
async def predict_crop(
    request: Request,
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...)
):
    """
    Handles crop prediction based on user input.
    """
    if model is None or scaler is None:
        predicted_crop = "Error: Model or scaler not loaded. Cannot make prediction."
        return templates.TemplateResponse("crop-prediction-results.html", {
            "request": request,
            "predicted_crop": predicted_crop,
            "error": True
        })

    try:
        # Create a numpy array from the input features
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale the input features using the loaded scaler
        scaled_features = scaler.transform(input_features)

        # Make prediction
        predicted_crop = model.predict(scaled_features)[0]

    except Exception as e:
        predicted_crop = f"An error occurred during prediction: {e}"
        return templates.TemplateResponse("crop-prediction-results.html", {
            "request": request,
            "predicted_crop": predicted_crop,
            "error": True
        })

    return templates.TemplateResponse("crop-prediction-results.html", {
        "request": request,
        "predicted_crop": predicted_crop,
        "error": False
    })



# --- Uvicorn Run ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
