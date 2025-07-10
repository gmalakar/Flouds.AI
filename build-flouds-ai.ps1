# =============================================================================
# File: build-flouds-ai.ps1
# Date: 2024-07-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
#
# This script builds and optionally pushes a Docker image for Flouds AI.
#
# Usage:
#   ./build-flouds-ai.ps1 [-ImageName <name>] [-Tag <tag>] [-GPU] [-PushImage] [-Force]
#
# Parameters:
#   -ImageName     : Base name of the Docker image (default: "gmalakar/flouds-ai-cpu")
#   -Tag           : Tag for the Docker image (default: "latest")
#   -GPU           : Build with GPU support (adds -gpu to the tag)
#   -PushImage     : Push the image to a Docker registry after building
#   -Force         : Force rebuild even if the image already exists
# =============================================================================

param (
    [string]$ImageName = "gmalakar/flouds-ai-cpu",
    [string]$Tag = "latest",
    [switch]$GPU = $false,
    [switch]$PushImage = $false,
    [switch]$Force = $false
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
    Write-Host "❌ ERROR: $Message" -ForegroundColor Red
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

# Add CPU/GPU suffix to tag if not already present
$cpuTag = if ($GPU) { "${Tag}-gpu" } else { "${Tag}-cpu" }
if ($Tag -match "-cpu$" -or $Tag -match "-gpu$") {
    $cpuTag = $Tag  # Keep original if it already has a suffix
}

Write-Host "========================================================="
Write-Host "                  FLOUDS AI BUILD SCRIPT                 " -ForegroundColor Cyan
Write-Host "========================================================="
Write-Host "Image Name     : $ImageName"
Write-Host "Tag            : $cpuTag"
Write-Host "Full Image     : ${ImageName}:${cpuTag}"
Write-Host "GPU Support    : $GPU"
Write-Host "Push to Registry: $PushImage"
Write-Host "Force Rebuild  : $Force"
Write-Host "========================================================="

# Check Docker installation
Write-StepHeader "Checking Docker installation"
Check-Docker

# Check for Dockerfile
Write-StepHeader "Validating build prerequisites"
if (-not (Test-Path "Dockerfile")) {
    Write-Error "Dockerfile not found in the current directory"
    exit 1
}
Write-Success "Dockerfile found"

# Check app directory
if (-not (Test-Path "app")) {
    Write-Error "app directory not found. This is required for building the image."
    exit 1
}
Write-Success "app directory found"

# Full image name with tag
$fullImageName = "${ImageName}:${cpuTag}"

# Check if image already exists
Write-StepHeader "Checking for existing images"
$imageExists = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -eq $fullImageName }
if ($imageExists) {
    if (-not $Force) {
        Write-Warning "Image $fullImageName already exists"
        $confirmation = Read-Host "Rebuild? (y/n)"
        if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
            Write-Host "Build cancelled by user." -ForegroundColor Yellow
            exit 0
        }
    } else {
        Write-Warning "Image $fullImageName already exists. Forcing rebuild as requested."
    }
} else {
    Write-Success "No existing image found with name $fullImageName"
}

# Build Docker image
Write-StepHeader "Building Docker image"
Write-Host "Building $fullImageName..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow

$buildStartTime = Get-Date

$buildArgs = @(
    "build",
    "--no-cache"
)

# Add GPU flag if requested
if ($GPU) {
    $buildArgs += "--build-arg", "GPU=true"
    Write-Host "Building with GPU support enabled" -ForegroundColor Yellow
}

$buildArgs += "-t", $fullImageName, "."

# Execute docker build command
try {
    & docker $buildArgs
    
    if ($LASTEXITCODE -eq 0) {
        $buildEndTime = Get-Date
        $buildDuration = $buildEndTime - $buildStartTime
        
        Write-Success "Docker image built successfully: $fullImageName"
        Write-Host "Build completed in $($buildDuration.Minutes)m $($buildDuration.Seconds)s" -ForegroundColor Green
        
        # Get image details
        $imageInfo = docker image inspect $fullImageName --format "{{.Size}}"
        if ($imageInfo) {
            $sizeInMB = [math]::Round($imageInfo / 1024 / 1024, 2)
            Write-Host "Image size: $sizeInMB MB" -ForegroundColor Cyan
        }
        
        # Push image if requested
        if ($PushImage) {
            Write-StepHeader "Pushing image to registry"
            Write-Host "Pushing $fullImageName to registry..." -ForegroundColor Yellow
            
            docker push $fullImageName
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Image pushed successfully to registry"
            } else {
                Write-Error "Failed to push image to registry"
                Write-Host "Make sure you're logged into Docker Hub with 'docker login'" -ForegroundColor Yellow
                exit 1
            }
        }
    } else {
        Write-Error "Failed to build Docker image"
        exit 1
    }
}
catch {
    Write-Error "Error building Docker image: $_"
    exit 1
}

# Show available images
Write-StepHeader "Available Flouds AI images"
docker images "${ImageName}*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

Write-Host "`n== Build Complete ==`n" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Run the container with: ./start-flouds-ai.ps1 -ImageName $fullImageName" -ForegroundColor Gray
if (-not $PushImage) {
    Write-Host "  2. Push to registry with: docker push $fullImageName" -ForegroundColor Gray
}
Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host "  .\build-flouds-ai.ps1" -ForegroundColor Gray
Write-Host "  .\build-flouds-ai.ps1 -Tag v1.0.0" -ForegroundColor Gray
Write-Host "  .\build-flouds-ai.ps1 -GPU" -ForegroundColor Gray
Write-Host "  .\build-flouds-ai.ps1 -PushImage" -ForegroundColor Gray
Write-Host "  .\build-flouds-ai.ps1 -Force" -ForegroundColor Gray
Write-Host ""