from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.preprocessing import clean_text
from typing import Optional
import os

# Set this BEFORE importing any tensorflow modules to use legacy Keras (Keras 2) with TF 2.16+
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Try to import multi-model utils, fallback to single model
try:
    from app.utils_multimodel import get_prediction, get_available_models
    MULTI_MODEL = True
except ImportError:
    from app.utils import get_prediction
    MULTI_MODEL = False

app = FastAPI(title="Sentiment Slang Analyzer", version="2.0")

# Enable CORS for all origins so Vercel can communicate with Render backend
# Note: You can change ["*"] to your specific Vercel URL in production for better security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

class TextInput(BaseModel):
    text: str
    model_version: Optional[str] = "version1"  # Default to version1

@app.get("/")
def home():
    """Redirect to static index page"""
    return RedirectResponse(url="/static/index.html")

@app.get("/models")
def list_models():
    """Get available model versions"""
    if MULTI_MODEL:
        return get_available_models()
    else:
        return {"version1": {"available": True, "metadata": {"description": "Single model mode"}}}

@app.post("/predict")
def predict_sentiment(data: TextInput):
    try:
        cleaned_text = clean_text(data.text)
        
        # Get prediction with model version
        if MULTI_MODEL:
            prediction_result = get_prediction(cleaned_text, model_version=data.model_version)
        else:
            prediction_result = get_prediction(cleaned_text)
        
        # Extract data from result
        sentiment = prediction_result.get('sentiment', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        model_version = prediction_result.get('model_version', data.model_version)
        model_info = prediction_result.get('model_info', {})
        
        return {
            "input_text": data.text,
            "cleaned_text": cleaned_text,
            "prediction": sentiment,
            "confidence": confidence,
            "model_version": model_version,
            "model_info": model_info
        }
    except Exception as e:
        print(f"Error in predict_sentiment: {e}")
        raise