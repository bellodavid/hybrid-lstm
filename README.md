# Aflatoxin Risk Prediction API

A FastAPI-based REST API for predicting aflatoxin contamination risk in Nigerian states using weather data and machine learning.

## Features

- **Health Check Endpoint**: Monitor API status and model availability
- **Risk Prediction**: Predict aflatoxin contamination risk based on location and date
- **Automatic Weather Data**: Fetches 30-day weather history from Open-Meteo API
- **State Mapping**: Maps Nigerian states to representative agro-ecological zones
- **Containerized Deployment**: Docker support for easy deployment

## API Endpoints

### Health Check
```
GET /health
```
Returns API health status and model availability.

### Root Information
```
GET /
```
Returns API information and available endpoints.

### Predict Aflatoxin Risk
```
POST /predict
```
Predicts aflatoxin contamination risk.

**Request Body:**
```json
{
  "location": "Lagos",
  "date": "2024-08-01"  // Optional, defaults to current date
}
```

**Response:**
```json
{
  "location": "Lagos",
  "representative_city": "Ikeja",
  "weather_period": "7/3/2024 to 8/1/2024",
  "predicted_risk": 0.3456,
  "risk_level": "Medium"
}
```

## Deployment Options

### 1. Local Development

```bash
# Clone and navigate to the project
cd aflatoxin_fastapi

# Run the startup script
./start.sh
```

The API will be available at `http://localhost:8000`

### 2. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t aflatoxin-api .
docker run -p 8000:8000 aflatoxin-api
```

### 3. Cloud Deployment

#### AWS ECS/Fargate
1. Push Docker image to ECR
2. Create ECS task definition
3. Deploy as Fargate service
4. Configure Application Load Balancer

#### Google Cloud Run
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/aflatoxin-api
gcloud run deploy --image gcr.io/PROJECT-ID/aflatoxin-api --platform managed
```

#### Azure Container Instances
```bash
# Deploy to Azure
az container create --resource-group myResourceGroup \
  --name aflatoxin-api --image myregistry.azurecr.io/aflatoxin-api:latest \
  --ports 8000 --dns-name-label aflatoxin-api
```

## Health Monitoring

The API includes built-in health checks:

- **Endpoint**: `GET /health`
- **Docker Health Check**: Automatic container health monitoring
- **Response Codes**: 
  - 200: Healthy (model loaded)
  - 503: Unhealthy (model not loaded)

## Supported Locations

The API supports all Nigerian states, mapped to representative cities:

- **North-West**: Kaduna, Kebbi zones
- **North-East**: Maiduguri zone
- **North-Central**: Abuja, Jos, Makurdi, Niger zones
- **South-West**: Ikeja zone
- **South-South**: Port Harcourt zone
- **South-East**: Port Harcourt, Makurdi zones

## Model Files Required

Ensure these files are in the `app/` directory:
- `aflatoxin_model.onnx` - ONNX model file
- `standard_scaler_params.json` - Scaler parameters
- `label_encoder_classes.json` - Label encoder classes

## Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"location": "Lagos", "date": "2024-08-01"}'
```

## Production Considerations

1. **Load Balancing**: Use multiple instances behind a load balancer
2. **Caching**: Implement Redis for weather data caching
3. **Monitoring**: Add logging and metrics collection
4. **Security**: Implement API authentication if needed
5. **Rate Limiting**: Add request rate limiting
6. **HTTPS**: Use SSL/TLS in production

## Troubleshooting

- **Model not loading**: Check if all model files are present
- **Weather API errors**: Verify internet connectivity
- **Location not found**: Ensure location is a valid Nigerian state
- **Health check failing**: Check model loading and dependencies