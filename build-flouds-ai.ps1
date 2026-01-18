# =============================================================================
# File: build-flouds-ai.ps1
# Date: 2024-07-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
#
# This script builds and optionally pushes a Docker image for Flouds AI.
#
# Usage:
#   ./build-flouds-ai.ps1 [-Tag <tag>] [-GPU] [-PushImage] [-Force] [-NoCache] [-Platform <arch>] [-AlsoTagLatest]
#
# Parameters:
#   -Tag           : Tag for the Docker image (default: "latest")
#   -GPU           : Build with GPU support (uses gmalakar/flouds-ai-gpu instead of gmalakar/flouds-ai-cpu)
#   -PushImage     : Push the image to a Docker registry after building
#   -Force         : Force rebuild even if the image already exists
#   -NoCache       : Disable Docker layer cache (slower but ensures clean build)
#   -Platform      : Target platform (e.g., "linux/amd64" or "linux/arm64")
#   -AlsoTagLatest : Also tag the image as 'latest' when using a custom tag
# =============================================================================

param (
    [string]$Tag = "latest",
    [switch]$GPU = $false,
    [switch]$PushImage = $false,
    [switch]$Force = $false,
    [switch]$NoCache = $false,
    [string]$Platform = "",
    [switch]$AlsoTagLatest = $false
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

function Test-Docker {
    try {
        $null = docker version 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker is not running or not accessible. Please start Docker and try again."
            exit 1
        }
        
        Write-Success "Docker is running"
        return $true
    }
    catch {
        Write-Error "Docker command failed: $_"
        exit 1
    }
}

# ========================== MAIN SCRIPT ==========================

# Set image name based on GPU flag
$ImageName = if ($GPU) { "gmalakar/flouds-ai-gpu" } else { "gmalakar/flouds-ai-cpu" }

# Full image name with tag
$fullImageName = "${ImageName}:${Tag}"

Write-Host "========================================================="
Write-Host "                  FLOUDS AI BUILD SCRIPT                 " -ForegroundColor Cyan
Write-Host "========================================================="
Write-Host "Image Name     : $ImageName"
Write-Host "Tag            : $Tag"
Write-Host "Full Image     : $fullImageName"
Write-Host "GPU Support    : $GPU"
Write-Host "Push to Registry: $PushImage"
Write-Host "Force Rebuild  : $Force"
Write-Host "Use Cache      : $(-not $NoCache)"
if ($Platform) {
    Write-Host "Platform       : $Platform"
}
if ($AlsoTagLatest) {
    Write-Host "Also Tag       : ${ImageName}:latest"
}
Write-Host "========================================================="

# Check Docker installation
Write-StepHeader "Checking Docker installation"
Test-Docker

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

# Check if image already exists
Write-StepHeader "Checking for existing images"
$imageExists = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -eq $fullImageName }
if ($imageExists) {
    if (-not $Force) {
        Write-Warning "$fullImageName already exists"
        $confirmation = Read-Host "Rebuild? (y/n)"
        if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
            Write-Host "Build cancelled by user." -ForegroundColor Yellow
            exit 0
        }
    }
    else {
        Write-Warning "$fullImageName already exists. Forcing rebuild as requested."
    }
}
else {
    Write-Success "No existing image found with name $fullImageName"
}

# Build Docker image
Write-StepHeader "Building Docker image"
Write-Host "Building $fullImageName..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow

$buildStartTime = Get-Date

# Enable BuildKit for faster builds
$env:DOCKER_BUILDKIT = "1"

$buildArgs = @(
    "build"
)

# Add cache flag only if explicitly requested
if ($NoCache) {
    $buildArgs += "--no-cache"
    Write-Host "Building without cache (clean build)" -ForegroundColor Yellow
}
else {
    Write-Host "Using Docker layer cache for faster builds" -ForegroundColor Yellow
}

# Add platform if specified, otherwise default to linux/amd64
if ($Platform) {
    $buildArgs += "--platform", $Platform
    Write-Host "Building for platform: $Platform" -ForegroundColor Yellow
}
else {
    $buildArgs += "--platform", "linux/amd64"
}

$buildArgs += @(
    "-t", $fullImageName,
    "."
)

# Add GPU flag if requested
if ($GPU) {
    $buildArgs += "--build-arg", "GPU=true"
    Write-Host "Building with GPU support enabled" -ForegroundColor Yellow
}

# Execute docker build command with retry logic
try {
    $maxRetries = 3
    $retryCount = 0
    $success = $false
    
    Write-Host "Starting Docker build with max $maxRetries attempts..." -ForegroundColor Yellow
    
    while (-not $success -and $retryCount -lt $maxRetries) {
        $retryCount++
        
        if ($retryCount -gt 1) {
            $waitTime = [math]::Pow(2, $retryCount - 1) * 10 # Exponential backoff: 10s, 20s, 40s...
            Write-Warning "Retrying in $waitTime seconds (Attempt $retryCount of $maxRetries)..."
            Start-Sleep -Seconds $waitTime
        }
        
        Write-Host "Docker build attempt $retryCount of $maxRetries..." -ForegroundColor Yellow
        & docker $buildArgs
        
        if ($LASTEXITCODE -eq 0) {
            $success = $true
        }
        else {
            if ($retryCount -lt $maxRetries) {
                Write-Warning "Docker build failed. Will retry..."
            }
        }
    }
    
    if ($success) {
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
        
        # Tag as latest if requested and tag is not already 'latest'
        if ($AlsoTagLatest -and $Tag -ne "latest") {
            $latestImageName = "${ImageName}:latest"
            Write-Host "Tagging as latest: $latestImageName" -ForegroundColor Yellow
            docker tag $fullImageName $latestImageName
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Also tagged as: $latestImageName"
            }
            else {
                Write-Warning "Failed to tag as latest"
            }
        }
        
        # Push image if requested
        if ($PushImage) {
            Write-StepHeader "Pushing image to registry"
            Write-Host "Pushing $fullImageName to registry..." -ForegroundColor Yellow
            
            # Also add retry for push operation
            $pushRetries = 2
            $pushSuccess = $false
            $pushAttempt = 0
            
            while (-not $pushSuccess -and $pushAttempt -lt $pushRetries) {
                $pushAttempt++
                if ($pushAttempt -gt 1) {
                    Write-Warning "Retrying push in 10 seconds (Attempt $pushAttempt of $pushRetries)..."
                    Start-Sleep -Seconds 10
                }
                
                docker push $fullImageName
                
                if ($LASTEXITCODE -eq 0) {
                    $pushSuccess = $true
                    Write-Success "Image pushed successfully: $fullImageName"
                    
                    # Push latest tag if it was created
                    if ($AlsoTagLatest -and $Tag -ne "latest") {
                        $latestImageName = "${ImageName}:latest"
                        Write-Host "Pushing latest tag: $latestImageName" -ForegroundColor Yellow
                        docker push $latestImageName
                        if ($LASTEXITCODE -eq 0) {
                            Write-Success "Latest tag pushed successfully"
                        }
                        else {
                            Write-Warning "Failed to push latest tag"
                        }
                    }
                }
                else {
                    Write-Warning "Push attempt $pushAttempt failed"
                }
            }
            
            if (-not $pushSuccess) {
                Write-Error "Failed to push image to registry after $pushRetries attempts"
                Write-Host "Make sure you're logged into Docker Hub with 'docker login'" -ForegroundColor Yellow
                exit 1
            }
        }
    }
    else {
        Write-Error "Failed to build Docker image after $maxRetries attempts"
        exit 1
    }
}
catch {
    Write-Error "Error building Docker image: $_"
    exit 1
}

# Show available images
Write-StepHeader "Available Flouds AI images"
docker images "gmalakar/flouds-ai*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

Write-Host "`n== Build Complete ==`n" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Run the container with: ./start-flouds-ai.ps1 -ImageName $ImageName -Tag $Tag" -ForegroundColor Gray
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