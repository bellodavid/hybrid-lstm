from pydantic import BaseModel, Field, validator
from typing import Optional

class AflatoxinRequest(BaseModel):
    location: str = Field(..., description="Nigerian state name (e.g., 'Lagos', 'Kaduna')")
    date: Optional[str] = Field(None, description="End date for 30-day weather period (YYYY-MM-DD format)")
    
    @validator('location')
    def validate_location(cls, v):
        valid_states = [
            'lagos', 'kaduna', 'kano', 'rivers', 'abuja', 'fct', 'ogun', 'oyo', 'katsina',
            'borno', 'bauchi', 'jigawa', 'kebbi', 'sokoto', 'zamfara', 'yobe', 'adamawa',
            'gombe', 'taraba', 'nasarawa', 'kwara', 'kogi', 'benue', 'plateau', 'niger',
            'osun', 'ondo', 'ekiti', 'akwa ibom', 'bayelsa', 'cross river', 'delta', 'edo',
            'imo', 'abia', 'anambra', 'enugu', 'ebonyi'
        ]
        if v.lower().strip() not in valid_states:
            raise ValueError(f'Invalid Nigerian state: {v}')
        return v.lower().strip()

class AflatoxinResponse(BaseModel):
    location: str
    representative_city: str
    weather_period: str
    predicted_risk: float
    risk_level: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str
    error: Optional[str] = None