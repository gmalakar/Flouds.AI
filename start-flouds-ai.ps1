# =============================================================================
# File: .ps1
# Date: 2024-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
#
# This script sets up and runs the Flouds AI container with proper volume mapping
# and environment variable handling based on a .env file.
#
# Usage:
#   ./start-flouds-ai.ps1 [-EnvFile <path>] [-InstanceName <name>] [-ImageName <name>] [-Force] [-BuildImage]
#
# Parameters:
#   -EnvFile       : Path to .env file (default: ".env")
#   -InstanceName  : Name of the Docker container (default: "flouds-ai-instance")
#   -ImageName     : Docker image to use (default: "gmalakar/flouds-ai-cpu:latest")
#   -Force         : Force restart container if it exists
#   -BuildImage    : Build Docker image locally before starting container
# =============================================================================

param (
    [string]$EnvFile = ".env",
    [string]$InstanceName = "flouds-ai-instance",
    [string]$ImageName = "gmalakar/flouds-ai-cpu:latest",
    [switch]$Force = $false,
    [switch]$BuildImage = $false
)

# ========================== HELPER FUNCTIONS ==========================

function Write-StepHeader {
    param ([string]$Message)
    Write-Host "`n== $Message ==" -ForegroundColor Cyan
}

function Write-Success {
    param ([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param ([string]$Message)
    Write-Host "⚠️ $Message" -ForegroundColor Yellow
}

function Write-Error {
    param ([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Ensure-Network {
    param ([string]$Name)
    
    if (-not (docker network ls --format '{{.Name}}' | Where-Object { $_ -eq $Name })) {
        Write-Host "🔧 Creating network: $Name" -ForegroundColor Yellow
        docker network create $Name | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Network $Name created successfully"
        } else {
            Write-Error "Failed to create network: $Name"
            exit 1
        }
    } else {
        Write-Success "Network $Name already exists"
    }
}

function Read-EnvFile {
    param ([string]$FilePath)
    
    $envVars = @{}
    if (Test-Path $FilePath) {
        Get-Content $FilePath | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                $key = $Matches[1].Trim()
                $value = $Matches[2].Trim()
                # Remove quotes if present
                $value = $value -replace '^"(.*)"$', '$1'
                $value = $value -replace "^'(.*)'$", '$1'
                $envVars[$key] = $value
            }
        }
    }
    return $envVars
}

function Check-Docker {
    try {
        $process = Start-Process -FilePath "docker" -ArgumentList "version" -NoNewWindow -Wait -PassThru -RedirectStandardError "NUL"
        if ($process.ExitCode -ne 0) {
            Write-Error "Docker is not running or not accessible. Please start Docker and try again."
            exit 1
        }
        Write-Success "Docker is running"
        return $true
    } catch {
        Write-Error "Docker command failed: $_"
        exit 1
    }
}

# ========================== MAIN SCRIPT ==========================

Write-Host "========================================================="
Write-Host "                 FLOUDS.PY STARTER SCRIPT                " -ForegroundColor Cyan
Write-Host "========================================================="
Write-Host "Instance Name : $InstanceName"
Write-Host "Image         : $ImageName"
Write-Host "Environment   : $EnvFile"
Write-Host "Build Image   : $BuildImage"
Write-Host "Force Restart : $Force"
Write-Host "========================================================="

# Check if Docker is available
Write-StepHeader "Checking Docker installation"
Check-Docker

# Read environment variables
Write-StepHeader "Reading environment configuration"
if (-not (Test-Path $EnvFile)) {
    Write-Error "$EnvFile not found. Please create this file with required environment variables."
    exit 1
} 
Write-Success "Using environment file: $EnvFile"
$envVars = Read-EnvFile -FilePath $EnvFile

# Validate required environment variables
Write-StepHeader "Validating required environment variables"

# Check ONNX config file
if (-not $envVars.ContainsKey("FLOUDS_ONNX_CONFIG_FILE_AT_HOST")) {
    Write-Error "FLOUDS_ONNX_CONFIG_FILE_AT_HOST environment variable is required but not set in $EnvFile"
    exit 1
}
$configPath = $envVars["FLOUDS_ONNX_CONFIG_FILE_AT_HOST"]
if (-not (Test-Path $configPath)) {
    Write-Error "ONNX config file not found: $configPath"
    Write-Error "This file is required. Please check the path and try again."
    exit 1
}
Write-Success "Found ONNX config file: $configPath"

# Check ONNX model path
if (-not $envVars.ContainsKey("FLOUDS_ONNX_MODEL_PATH_AT_HOST")) {
    Write-Warning "FLOUDS_ONNX_MODEL_PATH_AT_HOST not set. Container will use internal models only."
} else {
    $modelPath = $envVars["FLOUDS_ONNX_MODEL_PATH_AT_HOST"]
    if (-not (Test-Path $modelPath)) {
        Write-Warning "ONNX model path does not exist: $modelPath"
        Write-Warning "Creating directory..."
        try {
            New-Item -ItemType Directory -Path $modelPath -Force | Out-Null
            Write-Success "ONNX model directory created: $modelPath"
        } catch {
            Write-Error "Failed to create ONNX model directory: $_"
            exit 1
        }
    } else {
        Write-Success "Found ONNX model path: $modelPath"
    }
}

# Ensure log directory exists
if (-not $envVars.ContainsKey("FLOUDS_LOG_PATH_AT_HOST")) {
    Write-Warning "FLOUDS_LOG_PATH_AT_HOST not set. Container logs will not be persisted to host."
} else {
    $logPath = $envVars["FLOUDS_LOG_PATH_AT_HOST"]
    if (-not (Test-Path $logPath)) {
        Write-Warning "Log directory does not exist: $logPath"
        Write-Warning "Creating directory..."
        try {
            New-Item -ItemType Directory -Path $logPath -Force | Out-Null
            Write-Success "Log directory created: $logPath"
        } catch {
            Write-Error "Failed to create log directory: $_"
            exit 1
        }
    } else {
        Write-Success "Found log directory: $logPath"
    }
}

# Build image if requested
if ($BuildImage) {
    Write-StepHeader "Building Docker image"
    Write-Host "Building $ImageName..."
    docker build -t $ImageName .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build Docker image."
        exit 1
    }
    
    Write-Success "Docker image built successfully: $ImageName"
}

# Create Docker network
Write-StepHeader "Setting up Docker network"
$aiNetwork = "flouds_ai_network"
Ensure-Network -Name $aiNetwork

# Stop and remove existing container if it exists
Write-StepHeader "Managing container instance"
$containerExists = docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $InstanceName }
if ($containerExists) {
    Write-Warning "Container $InstanceName already exists"
    if ($Force) {
        Write-Host "Stopping and removing existing container: $InstanceName"
        docker stop $InstanceName | Out-Null
        docker rm $InstanceName | Out-Null
        Write-Success "Container removed"
    } else {
        Write-Error "Container already exists. Use -Force to replace it."
        exit 1
    }
}

# Prepare Docker run command
Write-StepHeader "Preparing container configuration"
$dockerArgs = @(
    "run", "-d", 
    "--name", $InstanceName, 
    "--network", $aiNetwork, 
    "-p", "19690:19690",
    "-e", "FLOUDS_API_ENV=Production", 
    "-e", "FLOUDS_DEBUG_MODE=0"
)

# Configure ONNX config file mapping
if ($envVars.ContainsKey("FLOUDS_ONNX_CONFIG_FILE_AT_HOST")) {
    $configPath = $envVars["FLOUDS_ONNX_CONFIG_FILE_AT_HOST"]
    $dockerConfigPath = $envVars["FLOUDS_ONNX_CONFIG_FILE"]
    if (-not $dockerConfigPath) {
        $dockerConfigPath = "/flouds-ai/app/config/onnx_config.json"
    }
    Write-Host "Mapping ONNX config: $configPath → $dockerConfigPath"
    $dockerArgs += "-v", "${configPath}:${dockerConfigPath}:ro"
    $dockerArgs += "-e", "FLOUDS_ONNX_CONFIG_FILE=${dockerConfigPath}"
}

# Configure ONNX model directory mapping
if ($envVars.ContainsKey("FLOUDS_ONNX_MODEL_PATH_AT_HOST")) {
    $modelPath = $envVars["FLOUDS_ONNX_MODEL_PATH_AT_HOST"] 
    $dockerOnnxPath = $envVars["FLOUDS_ONNX_ROOT"]
    if (-not $dockerOnnxPath) {
        $dockerOnnxPath = "/flouds-ai/onnx"
    }
    Write-Host "Mapping ONNX models: $modelPath → $dockerOnnxPath"
    $dockerArgs += "-v", "${modelPath}:${dockerOnnxPath}:ro"
    $dockerArgs += "-e", "FLOUDS_ONNX_ROOT=${dockerOnnxPath}"
}

# Configure log directory mapping
if ($envVars.ContainsKey("FLOUDS_LOG_PATH_AT_HOST")) {
    $logPath = $envVars["FLOUDS_LOG_PATH_AT_HOST"]
    $dockerLogPath = $envVars["FLOUDS_LOG_PATH"] 
    if (-not $dockerLogPath) {
        $dockerLogPath = "/var/log/flouds"
    }
    Write-Host "Mapping logs: $logPath → $dockerLogPath"
    $dockerArgs += "-v", "${logPath}:${dockerLogPath}:rw"
    $dockerArgs += "-e", "FLOUDS_LOG_PATH=${dockerLogPath}"
}

# Add image name
$dockerArgs += $ImageName

# Start the container
Write-StepHeader "Starting Flouds.Py container"
Write-Host "Command: docker $($dockerArgs -join ' ')" -ForegroundColor Gray

try {
    & docker $dockerArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Flouds.Py container started successfully"
        Write-Success "API available at: http://localhost:19690/docs"
        
        # Wait for container to initialize
        Write-Host "Waiting for container to initialize..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        
        # Show container status
        Write-StepHeader "Container Status"
        docker ps --filter "name=$InstanceName" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    } else {
        Write-Error "Failed to start Flouds.Py container."
        exit 1
    }
}
catch {
    Write-Error "Error starting Flouds.Py container: $_"
    exit 1
}

# Show logs option
Write-StepHeader "Container Management"
Write-Host "Use the following commands to manage the container:" -ForegroundColor Cyan
Write-Host "  * View logs: docker logs -f $InstanceName" -ForegroundColor Gray
Write-Host "  * Stop container: docker stop $InstanceName" -ForegroundColor Gray
Write-Host "  * Remove container: docker rm $InstanceName" -ForegroundColor Gray
Write-Host ""

$showLogs = Read-Host "Would you like to view container logs now? (y/n)"
if ($showLogs -eq "y" -or $showLogs -eq "Y") {
    docker logs -f $InstanceName
}