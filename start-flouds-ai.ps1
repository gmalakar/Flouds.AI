# =============================================================================
# File: start-flouds-ai.ps1
# Date: 2024-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
#
# This script sets up and runs the Flouds AI container with proper volume mapping
# and environment variable handling based on a .env file.
#
# Usage:
#   ./start-flouds-ai.ps1 [-EnvFile <path>] [-InstanceName <name>] [-ImageName <name>] [-Tag <tag>] [-Force] [-BuildImage] [-PullAlways]
#
# Parameters:
#   -EnvFile       : Path to .env file (default: ".env")
#   -InstanceName  : Name of the Docker container (default: "flouds-ai-instance")
#   -ImageName     : Base Docker image name (default: "gmalakar/flouds-ai-cpu")
#   -Tag           : Tag for the Docker image (default: "latest")
#   -GPU           : Use GPU image (default: $false)
#   -Force         : Force restart container if it exists
#   -BuildImage    : Build Docker image locally before starting container
#   -PullAlways    : Always pull image from registry before running
# =============================================================================

param (
    [string]$EnvFile = ".env",
    [string]$InstanceName = "flouds-ai-instance",
    [string]$ImageName = "gmalakar/flouds-ai-cpu",
    [string]$Tag = "latest",
    [switch]$GPU = $false,
    [switch]$Force = $false,
    [switch]$BuildImage = $false,
    [switch]$PullAlways = $false
)

# ========================== HELPER FUNCTIONS ==========================

function Write-StepHeader {
    param ([string]$Message)
    Write-Host "`n== $Message ==" -ForegroundColor Cyan
}

function Write-Success {
    param ([string]$Message)
    Write-Host " $Message" -ForegroundColor Green
}

function Write-Warning {
    param ([string]$Message)
    Write-Host " $Message" -ForegroundColor Yellow
}

function Write-Error {
    param ([string]$Message)
    Write-Host " $Message" -ForegroundColor Red
    exit 1
}

function New-NetworkIfMissing {
    param ([string]$Name)
    if (-not (docker network ls --format '{{.Name}}' | Where-Object { $_ -eq $Name })) {
        Write-Host " Creating network: $Name" -ForegroundColor Yellow
        docker network create $Name | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Network $Name created successfully"
        } else {
            Write-Error "Failed to create network: $Name"
        }
    } else {
        Write-Success "Network $Name already exists"
    }
}

function Connect-NetworkIfNotConnected {
    param (
        [string]$Container,
        [string]$Network
    )
    $containerRunning = docker ps --format '{{.Names}}' | Where-Object { $_ -eq $Container }
    if (-not $containerRunning) {
        Write-Warning "Container $Container is not running. Skipping network connection."
        return
    }
    $networks = docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' $Container
    if ($networks -notmatch "\b$Network\b") {
        Write-Host " Connecting network $Network to container $Container" -ForegroundColor Yellow
        docker network connect $Network $Container 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Successfully connected $Container to $Network"
        } else {
            Write-Warning "Failed to connect $Container to $Network"
        }
    } else {
        Write-Success "Container $Container is already connected to $Network"
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
                $value = $value -replace '^"(.*)"$', '$1'
                $value = $value -replace "^'(.*)'$", '$1'
                $envVars[$key] = $value
            }
        }
    }
    return $envVars
}

function Test-Docker {
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

function Test-DirectoryWritable {
    param ([string]$Path)
    try {
        $testFile = Join-Path -Path $Path -ChildPath "test_write_$([Guid]::NewGuid().ToString()).tmp"
        [System.IO.File]::WriteAllText($testFile, "test")
        Remove-Item -Path $testFile -Force
        return $true
    } catch {
        return $false
    }
}

function Set-DirectoryPermissions {
    param (
        [string]$Path,
        [string]$Description
    )
    if (-not (Test-Path $Path)) {
        Write-Warning "$Description directory does not exist: $Path"
        Write-Host "Creating directory..." -ForegroundColor Yellow
        try {
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
            Write-Success "$Description directory created: $Path"
        } catch {
            Write-Error "Failed to create $Description directory: $_"
            exit 1
        }
    } else {
        Write-Success "Found $Description directory: $Path"
    }
    
    # Test if directory is writable
    if (Test-DirectoryWritable -Path $Path) {
        Write-Success "$Description directory is writable: $Path"
    } else {
        Write-Warning "$Description directory is not writable: $Path"
        Write-Host "Setting permissions on $Description directory..." -ForegroundColor Yellow
        try {
            # Try to set permissions (works on Windows)
            $acl = Get-Acl $Path
            $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Everyone", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
            $acl.SetAccessRule($accessRule)
            Set-Acl $Path $acl
            Write-Success "Permissions set successfully"
        } catch {
            Write-Warning "Failed to set permissions: $_"
            Write-Warning "$Description may not be writable. Please check directory permissions manually."
            $continue = Read-Host "Continue anyway? (y/n)"
            if ($continue -ne "y") {
                Write-Host "Aborted by user." -ForegroundColor Red
                exit 0
            }
        }
    }
}

# ========================== MAIN SCRIPT ==========================

# Adjust image name for GPU
if ($GPU -and $ImageName -eq "gmalakar/flouds-ai-cpu") {
    $ImageName = "gmalakar/flouds-ai-gpu"
}
$fullImageName = "${ImageName}:${Tag}"

$workingDir = "/flouds-ai"

Write-Host "========================================================="
Write-Host "                 Flouds AI STARTER SCRIPT                " -ForegroundColor Cyan
Write-Host "========================================================="
Write-Host "Instance Name : $InstanceName"
Write-Host "Base Image    : $ImageName"
Write-Host "Tag           : $Tag"
Write-Host "Full Image    : $fullImageName"
Write-Host "Environment   : $EnvFile"
Write-Host "Build Image   : $BuildImage"
Write-Host "Force Restart : $Force"
Write-Host "Pull Always   : $PullAlways"
Write-Host "========================================================="

Write-StepHeader "Checking Docker installation"
Test-Docker

Write-StepHeader "Reading environment configuration"
if (-not (Test-Path $EnvFile)) {
    Write-Error "$EnvFile not found. Please create this file with required environment variables."
    exit 1
}
Write-Success "Using environment file: $EnvFile"
$envVars = Read-EnvFile -FilePath $EnvFile

Write-StepHeader "Validating required environment variables"
if (-not $envVars.ContainsKey("FLOUDS_ONNX_CONFIG_FILE_AT_HOST")) {
    Write-Error "FLOUDS_ONNX_CONFIG_FILE_AT_HOST environment variable is required but not set in $EnvFile"
    exit 1
}
$configPath = $envVars["FLOUDS_ONNX_CONFIG_FILE_AT_HOST"]
if (-not (Test-Path $configPath)) {
    Write-Error "ONNX config file not found: $configPath"
    exit 1
}
Write-Success "Found ONNX config file: $configPath"

if (-not $envVars.ContainsKey("FLOUDS_ONNX_MODEL_PATH_AT_HOST")) {
    Write-Error "FLOUDS_ONNX_MODEL_PATH_AT_HOST environment variable is required but not set in $EnvFile"
    exit 1
}
$modelPath = $envVars["FLOUDS_ONNX_MODEL_PATH_AT_HOST"]
if (-not (Test-Path $modelPath)) {
    Write-Error "ONNX model path does not exist: $modelPath"
    exit 1
}
Write-Success "Found ONNX model path: $modelPath"

# Check and set permissions for log directory
if ($envVars.ContainsKey("FLOUDS_LOG_PATH_AT_HOST")) {
    $logPath = $envVars["FLOUDS_LOG_PATH_AT_HOST"]
    Set-DirectoryPermissions -Path $logPath -Description "Log"
} else {
    Write-Warning "FLOUDS_LOG_PATH_AT_HOST not set. Container logs will not be persisted to host."
}

if ($BuildImage) {
    Write-StepHeader "Building Docker image"
    Write-Host "Building $fullImageName..." -ForegroundColor Yellow
    docker build -t $fullImageName .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build Docker image."
        exit 1
    }
    Write-Success "Docker image built successfully: $fullImageName"
}

Write-StepHeader "Setting up Docker network"
$aiNetwork = "flouds_ai_network"
New-NetworkIfMissing -Name $aiNetwork

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

Write-StepHeader "Preparing container configuration"
$dockerArgs = @("run", "-d")
if ($PullAlways) {
    $dockerArgs += "--pull"
    $dockerArgs += "always"
}
$dockerArgs += @(
    "--name", $InstanceName,
    "--network", $aiNetwork,
    "-p", "19690:19690",
    "-e", "FLOUDS_API_ENV=Production",
    "-e", "FLOUDS_DEBUG_MODE=0"
)

# ONNX config file mapping
if ($envVars.ContainsKey("FLOUDS_ONNX_CONFIG_FILE_AT_HOST")) {
    $dockerConfigPath = "$workingDir/app/config/onnx_config.json"
    Write-Host "Mapping ONNX config: $configPath  $dockerConfigPath"
    $dockerArgs += "-v"
    $dockerArgs += "${configPath}:${dockerConfigPath}:ro"
}

# ONNX model directory mapping
if ($envVars.ContainsKey("FLOUDS_ONNX_MODEL_PATH_AT_HOST")) {
    $dockerOnnxPath = "$workingDir/onnx"
    Write-Host "Mapping ONNX models: $modelPath  $dockerOnnxPath"
    $dockerArgs += "-v"
    $dockerArgs += "${modelPath}:${dockerOnnxPath}:ro"
}

# Log directory mapping
if ($envVars.ContainsKey("FLOUDS_LOG_PATH_AT_HOST")) {
    $dockerLogPath = "$workingDir/logs"
    Write-Host "Mapping logs: $logPath  $dockerLogPath"
    $dockerArgs += "-v"
    $dockerArgs += "${logPath}:${dockerLogPath}:rw"
    $dockerArgs += "-e"
    $dockerArgs += "FLOUDS_LOG_PATH=$dockerLogPath"
}

# Add platform flag if specified
if ($envVars.ContainsKey("DOCKER_PLATFORM")) {
    $platform = $envVars["DOCKER_PLATFORM"]
    Write-Host "Setting platform: $platform"
    $dockerArgs += "--platform"
    $dockerArgs += $platform
}

$dockerArgs += $fullImageName

Write-StepHeader "Starting Flouds AI container"
Write-Host "Command: docker $($dockerArgs -join ' ')" -ForegroundColor Gray

try {
    & docker $dockerArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Flouds AI container started successfully"
        Write-Success "API available at: http://localhost:19690/docs"
        Write-Host "Waiting for container to initialize..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3

        Write-StepHeader "Container Status"
        docker ps --filter "name=$InstanceName" --format "table {{.ID}}`t{{.Image}}`t{{.Status}}`t{{.Ports}}"
    } else {
        Write-Error "Failed to start Flouds AI container."
        exit 1
    }
}
catch {
    Write-Error "Error starting Flouds AI container: $_"
    exit 1
}

Write-StepHeader "Container Management"
Write-Host "Use the following commands to manage the container:" -ForegroundColor Cyan
Write-Host "  * View logs: docker logs -f $InstanceName" -ForegroundColor Gray
Write-Host "  * Stop container: docker stop $InstanceName" -ForegroundColor Gray
Write-Host "  * Remove container: docker rm $InstanceName" -ForegroundColor Gray
Write-Host "  * View API docs: http://localhost:19690/docs" -ForegroundColor Gray
Write-Host ""

$showLogs = Read-Host "Would you like to view container logs now? (y/n)"
if ($showLogs -eq "y" -or $showLogs -eq "Y") {
    docker logs -f $InstanceName
}
