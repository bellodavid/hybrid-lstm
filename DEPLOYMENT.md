# Deployment Guide

## Quick Start

1. **Local Testing**:
   ```bash
   ./start.sh
   python test_api.py
   ```

2. **Docker**:
   ```bash
   docker-compose up --build
   ```

3. **Health Check**: Visit `http://localhost:8000/health`

## Cloud Deployment Options

### AWS (Recommended for Production)

#### Option 1: AWS App Runner (Easiest)
```bash
# Build and push to ECR
aws ecr create-repository --repository-name aflatoxin-api
docker build -t aflatoxin-api .
docker tag aflatoxin-api:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/aflatoxin-api:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/aflatoxin-api:latest

# Create App Runner service via AWS Console
# - Source: Container registry
# - Image URI: Your ECR image
# - Port: 8000
# - Health check: /health
```

#### Option 2: ECS Fargate
```bash
# Create task definition and service via AWS Console or CLI
# Configure Application Load Balancer with health checks
```

### Google Cloud Platform

#### Cloud Run (Serverless)
```bash
# Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Build and deploy
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/aflatoxin-api
gcloud run deploy aflatoxin-api \
  --image gcr.io/YOUR-PROJECT-ID/aflatoxin-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

### Microsoft Azure

#### Container Instances
```bash
# Create resource group
az group create --name aflatoxin-rg --location eastus

# Deploy container
az container create \
  --resource-group aflatoxin-rg \
  --name aflatoxin-api \
  --image your-registry/aflatoxin-api:latest \
  --ports 8000 \
  --dns-name-label aflatoxin-api-unique \
  --restart-policy Always
```

### DigitalOcean

#### App Platform
1. Connect your GitHub repository
2. Configure build settings:
   - Build command: `docker build -t aflatoxin-api .`
   - Run command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. Set environment variables if needed
4. Deploy

### Heroku

```bash
# Install Heroku CLI and login
heroku create aflatoxin-api-your-name

# Set stack to container
heroku stack:set container -a aflatoxin-api-your-name

# Deploy
git push heroku main
```

## Production Configuration

### Environment Variables
```bash
# Optional configurations
export PORT=8000
export HOST=0.0.0.0
export LOG_LEVEL=info
```

### Load Balancer Health Check
- **Path**: `/health`
- **Port**: 8000
- **Protocol**: HTTP
- **Interval**: 30 seconds
- **Timeout**: 5 seconds
- **Healthy threshold**: 2
- **Unhealthy threshold**: 3

### Monitoring Setup
```bash
# Example CloudWatch alarms (AWS)
aws cloudwatch put-metric-alarm \
  --alarm-name "aflatoxin-api-health" \
  --alarm-description "API health check" \
  --metric-name HealthyHostCount \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 60 \
  --threshold 1 \
  --comparison-operator LessThanThreshold
```

## Security Considerations

1. **API Authentication**: Add JWT or API key authentication
2. **Rate Limiting**: Implement request throttling
3. **CORS**: Configure appropriate CORS policies
4. **HTTPS**: Always use SSL/TLS in production
5. **Secrets**: Store model files and credentials securely

## Scaling

- **Horizontal**: Multiple container instances behind load balancer
- **Vertical**: Increase CPU/memory per container
- **Auto-scaling**: Configure based on CPU/memory usage
- **Caching**: Add Redis for weather data caching

## Cost Optimization

- **AWS**: Use Fargate Spot for non-critical workloads
- **GCP**: Cloud Run scales to zero when not in use
- **Azure**: Use consumption-based pricing
- **Caching**: Reduce external API calls with Redis