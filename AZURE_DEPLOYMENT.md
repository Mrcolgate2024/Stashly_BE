# Azure Deployment Guide for Stashly Financial Coach

## Prerequisites
- Azure account
- Docker Desktop installed (https://www.docker.com/products/docker-desktop)
- Azure CLI installed (https://aka.ms/installazurecli)
- WSL 2 enabled (if using Windows)

## Project Structure
The application is organized into several key directories:
- `Agents/`: Contains all AI agents including the refactored research agent system
- `Knowledge_base/`: Stores agent personas and configuration files
- `Market_data/`: Contains market data files and analysis tools
- Root level: Main application files and configuration

## Initial Setup

1. **Verify Installations**
```bash
# Check Docker installation
docker --version

# Check Azure CLI installation
az --version

# Login to Azure
az login
```

2. **Create Azure Resources**
```bash
# Create resource group
az group create --name Stashly --location westeurope

# Create Azure Container Registry (ACR)
az acr create --resource-group Stashly --name stashlybackend --sku Basic

# Enable Admin Access for ACR
az acr update --name stashlybackend --admin-enabled true

# Create App Service Plan
az appservice plan create --name StashlyAppServicePlan --resource-group Stashly --sku B1 --is-linux

# Create Web App
az webapp create --resource-group Stashly --plan StashlyAppServicePlan --name stashlybackendapp --deployment-container-image-name stashlybackend.azurecr.io/mybackend:latest
```

## Building and Deploying

1. **Login to Azure Container Registry**
```bash
az acr login --name stashlybackend
```

2. **Build and Push Docker Image**
```bash
# Build the image
docker build -t stashlybackend.azurecr.io/mybackend:latest .

# Push to Azure Container Registry
docker push stashlybackend.azurecr.io/mybackend:latest

# Verify image in ACR
az acr repository show-tags --name stashlybackend --repository mybackend --output table
```

3. **Configure Web App**
```bash
# Set Web App to use ACR image
az webapp config container set \
  --name stashlybackendapp \
  --resource-group Stashly \
  --docker-custom-image-name stashlybackend.azurecr.io/mybackend:latest \
  --docker-registry-server-url https://stashlybackend.azurecr.io \
  --docker-registry-server-user $(az acr credential show --name stashlybackend --query "username" --output tsv) \
  --docker-registry-server-password $(az acr credential show --name stashlybackend --query "passwords[0].value" --output tsv)
```

4. **Configure Environment Variables**
In Azure Portal:
- Navigate to App Service > Configuration > Application settings
- Add the following settings:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `TAVILY_API_KEY`: Your Tavily API key
  - `GOOGLE_API_KEY`: Your Google API key
  - `SIMLI_API_KEY`: Your Simli API key
  - `SIMLI_API_SECRET`: Your Simli API secret
  - `MARKET_DATA_PATH`: Path to market data files (default: /app/Market_data)

## Testing and Debugging

1. **Get Web App URL**
```bash
az webapp show --resource-group Stashly --name stashlybackendapp --query defaultHostName -o tsv
```

2. **Test API**
```bash
curl -X GET https://stashlybackendapp.azurewebsites.net/test
```

3. **View Logs**
```bash
az webapp log tail --name stashlybackendapp --resource-group Stashly
```

4. **Check Environment Variables**
```bash
az webapp config appsettings list --name stashlybackendapp --resource-group Stashly --output table
```

## Updating and Redeploying

When making changes to your code:

1. **Rebuild and Push Docker Image**
```bash
docker build -t stashlybackend.azurecr.io/mybackend:latest .
docker push stashlybackend.azurecr.io/mybackend:latest
```

2. **Restart Web App**
```bash
az webapp restart --name stashlybackendapp --resource-group Stashly
```

## Troubleshooting
- Check App Service logs for any issues
- Verify environment variables are set correctly
- Ensure Docker container is running with `docker ps`
- Check Azure Portal for deployment status and logs
- Verify market data files are properly mounted in the container
- Check research agent logs for any initialization issues 