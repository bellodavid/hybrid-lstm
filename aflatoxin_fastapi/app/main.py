from fastapi import FastAPI, HTTPException
from app.models import AflatoxinRequest, AflatoxinResponse, HealthResponse
from app.inference import AflatoxinPredictor
from app.middleware import RateLimitMiddleware
import os
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aflatoxin Risk Prediction API",
    description="API for predicting aflatoxin contamination risk in Nigerian states",
    version="1.0.0"
)

# Add rate limiting
app.add_middleware(RateLimitMiddleware, calls=100, period=3600)

# Initialize predictor
predictor = AflatoxinPredictor()

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessors on startup"""
    try:
        logger.info("Loading model on startup...")
        predictor.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_loaded = predictor.is_model_loaded()
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_loaded=model_loaded,
            version="1.0.0"
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_loaded=False,
            version="1.0.0",
            error=str(e)
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Aflatoxin Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.post("/predict", response_model=AflatoxinResponse)
async def predict_aflatoxin_risk(request: AflatoxinRequest):
    """Predict aflatoxin contamination risk"""
    try:
        if not predictor.is_model_loaded():
            logger.error("Model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"Prediction request for {request.location}")
        result = predictor.predict(location=request.location, date=request.date)
        if result is None:
            logger.error("Prediction failed")
            raise HTTPException(status_code=400, detail="Prediction failed")
        
        logger.info(f"Prediction successful: {result.predicted_risk}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")