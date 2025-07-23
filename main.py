from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import pickle
import joblib
import warnings
import uvicorn

warnings.filterwarnings('ignore')


app = FastAPI()


df = pd.read_csv("Crop_recommendation.csv")

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y = df['label'] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)




# Display unique crop labels to check for imbalance
print("Unique Crops and their counts:\n", y.value_counts())

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")  


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/index", response_class=HTMLResponse)
async def read_rt(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/recommendation", response_class=HTMLResponse)
async def recommendation(request: Request):
    return templates.TemplateResponse("recommendation.html", {"request": request})

@app.get("/soil-analysis", response_class=HTMLResponse)
async def soil_analysis(request: Request):
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



#***************************************************************************************

try:
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("⚠️ model.pkl not found! Prediction will not work.")

model = None
try:
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    print("✅ model.joblib loaded successfully for crop prediction.")
except FileNotFoundError:
    model = None
    print("⚠️ model.joblib not found! Crop prediction feature will not work.")
except Exception as e:
    model = None
    print(f"❌ Error loading model.joblib: {e}. Crop prediction feature will not work.")

@app.get("/crop-prediction", response_class=HTMLResponse)
async def get_crop_prediction_form(request: Request):
    return templates.TemplateResponse("crop-prediction-form.html", {"request": request})



#***************************************************************************************


@app.post("/analyze-soil", response_class=HTMLResponse)
async def analyze_soil(
    request: Request,
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    ph: float = Form(...)
):
    # Define optimal ranges for soil nutrients and pH
    optimal_ranges = {
        "nitrogen": {"min": 40, "max": 80},     # mg/kg
        "phosphorus": {"min": 20, "max": 50},   # mg/kg
        "potassium": {"min": 100, "max": 300},  # mg/kg
        "ph": {"min": 6.0, "max": 7.5}
    }

    # Perform analysis
    analysis_results = {
        "nitrogen_status": "Optimal" if optimal_ranges["nitrogen"]["min"] <= nitrogen <= optimal_ranges["nitrogen"]["max"] else "Needs Attention",
        "phosphorus_status": "Optimal" if optimal_ranges["phosphorus"]["min"] <= phosphorus <= optimal_ranges["phosphorus"]["max"] else "Needs Attention",
        "potassium_status": "Optimal" if optimal_ranges["potassium"]["min"] <= potassium <= optimal_ranges["potassium"]["max"] else "Needs Attention",
        "ph_status": "Optimal" if optimal_ranges["ph"]["min"] <= ph <= optimal_ranges["ph"]["max"] else "Needs Attention",
        "ph_recommendation": "", # Added for specific PH recommendations
        "nitrogen_recommendation": "", # Added for specific Nitrogen recommendations
        "phosphorus_recommendation": "", # Added for specific Phosphorus recommendations
        "potassium_recommendation": "" # Added for specific Potassium recommendations

    }

    # Determine overall soil health status
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

    # Add specific recommendations based on deviations
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
        "analysis_results": analysis_results # Pass the detailed analysis
    })




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
