# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
# ... existing imports ...
import requests
import indiapins
import datetime

app = FastAPI()

# ... (your existing app.mount and templates setup) ...

# NEW ENDPOINT: Data for Crop Recommendations by State
@app.get('/api/crop_by_state')
async def get_crop_by_state_data():
    # This is dummy data. In a real application, this would come from a database,
    # a more sophisticated recommendation model, or a static data file.
    crop_data = {
        "Andhra Pradesh": ["Rice", "Tobacco", "Sugarcane"],
        "Arunachal Pradesh": ["Rice", "Maize", "Millet"],
        "Assam": ["Tea", "Rice", "Jute"],
        "Bihar": ["Rice", "Wheat", "Maize"],
        "Chhattisgarh": ["Rice", "Maize", "Pulses"],
        "Goa": ["Rice", "Cashew", "Coconut"],
        "Gujarat": ["Cotton", "Groundnut", "Tobacco"],
        "Haryana": ["Wheat", "Rice", "Sugarcane"],
        "Himachal Pradesh": ["Apple", "Potato", "Maize"],
        "Jharkhand": ["Rice", "Maize", "Pulses"],
        "Karnataka": ["Coffee", "Sugarcane", "Rice"],
        "Kerala": ["Rubber", "Spices", "Coconut"],
        "Madhya Pradesh": ["Soybean", "Wheat", "Gram"],
        "Maharashtra": ["Sugarcane", "Cotton", "Jowar"],
        "Manipur": ["Rice", "Maize", "Potato"],
        "Meghalaya": ["Rice", "Maize", "Potato"],
        "Mizoram": ["Rice", "Maize", "Ginger"],
        "Nagaland": ["Rice", "Maize", "Pulses"],
        "Odisha": ["Rice", "Pulses", "Oilseeds"],
        "Punjab": ["Wheat", "Rice", "Cotton"],
        "Rajasthan": ["Bajra", "Wheat", "Mustard"],
        "Sikkim": ["Cardamom", "Ginger", "Oranges"],
        "Tamil Nadu": ["Sugarcane", "Rice", "Groundnut"],
        "Telangana": ["Rice", "Cotton", "Maize"],
        "Tripura": ["Rice", "Jute", "Tea"],
        "Uttar Pradesh": ["Wheat", "Sugarcane", "Rice"],
        "Uttarakhand": ["Rice", "Wheat", "Maize"],
        "West Bengal": ["Rice", "Jute", "Tea"]
    }
    return crop_data # FastAPI automatically converts dictionary to JSON

# ... (your existing routes like '/', '/get_weather', etc.) ...