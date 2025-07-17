#!/bin/bash
# =============================================================================
# File: start-flouds-ai.sh
# Date: 2024-07-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
#
# This script sets up and runs the Flouds AI container with proper volume mapping
# and environment variable handling based on a .env file.
#
# Usage:
#   ./start-flouds-ai.sh [--env-file <path>] [--instance <name>] [--image <name>] [--tag <tag>] [--gpu] [--force] [--build]
#
# Parameters:
#   --env-file     : Path to .env file (default: ".env")
#   --instance     : Name of the Docker container (default: "flouds-ai-instance")
#   --image        : Base Docker image name (default: "gmalakar/flouds-ai-cpu")
#   --tag          : Tag for the Docker image (default: "latest")
#   --gpu          : Use GPU image instead of CPU image
#   --force        : Force restart container if it exists
#   --build        : Build Docker image locally before starting container
#   --pull-always  : Always pull image from registry before running
#   --development  : Run in development mode
# =============================================================================

# Default parameters
ENV_FILE=".env"
INSTANCE_NAME="flouds-ai-instance"
IMAGE_NAME="gmalakar/flouds-ai-cpu"
TAG="latest"
USE_GPU=false
FORCE=false
BUILD_IMAGE=false
AI_NETWORK="flouds_ai_network"
PORT=19690
PULL_ALWAYS=false
DEVELOPMENT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --instance)
      INSTANCE_NAME="$2"
      shift 2
      ;;
    --image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --gpu)
      USE_GPU=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --build)
      BUILD_IMAGE=true
      shift
      ;;
    --pull-always)
      PULL_ALWAYS=true
      shift
      ;;
    --development)
      DEVELOPMENT=true
      shift
      ;;
    *)
      echo "❌ Unknown option: $1"
      exit 1
      ;;
  esac
done

# Adjust image name for GPU if needed
if [ "$USE_GPU" = true ] && [ "$IMAGE_NAME" = "gmalakar/flouds-ai-cpu" ]; then
  IMAGE_NAME="gmalakar/flouds-ai-gpu"
fi

# Create full image name with tag
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "========================================================="
echo "                 FLOUDS.AI STARTER SCRIPT                "
echo "========================================================="
echo "Instance Name : $INSTANCE_NAME"
echo "Base Image    : $IMAGE_NAME"
echo "Tag           : $TAG"
echo "Full Image    : $FULL_IMAGE_NAME"
echo "Environment   : $ENV_FILE"
echo "GPU Support   : $USE_GPU"
echo "Build Image   : $BUILD_IMAGE"
echo "Force Restart : $FORCE"
echo "Development   : $DEVELOPMENT"
echo "========================================================="

# Check Docker
echo
echo "== Checking Docker =="
if ! docker version >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi
echo "✅ Docker is running"

# Read .env file
echo
echo "== Reading environment configuration =="
if [[ ! -f "$ENV_FILE" ]]; then
    echo "⚠️ $ENV_FILE not found. Using default configuration."
else
    echo "✅ Using environment file: $ENV_FILE"
    # Convert to Unix line endings and source
    awk '{ sub("\r$", ""); print }' "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
    source "$ENV_FILE"
fi

# Validate required ONNX config file
echo
echo "== Validating ONNX configuration =="
if [[ -z "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST" ]]; then
    echo "❌ FLOUDS_ONNX_CONFIG_FILE_AT_HOST is required in $ENV_FILE"
    exit 1
fi

if [[ ! -f "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST" ]]; then
    echo "❌ ONNX config file not found: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST"
    exit 1
fi
echo "✅ Found ONNX config file: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST"

# Build image if requested
if [ "$BUILD_IMAGE" = true ]; then
    echo
    echo "== Building Docker image =="
    BUILD_ARGS=()
    if [ "$USE_GPU" = true ]; then
        BUILD_ARGS+=(--build-arg GPU=true)
    fi
    
    if docker build "${BUILD_ARGS[@]}" -t "$FULL_IMAGE_NAME" .; then
        echo "✅ Docker image built successfully: $FULL_IMAGE_NAME"
    else
        echo "❌ Failed to build Docker image."
        exit 1
    fi
fi

# Stop and remove existing container if it exists
echo
echo "== Managing container instance =="
if docker ps -a --format '{{.Names}}' | grep -q "^${INSTANCE_NAME}$"; then
    echo "⚠️ Container $INSTANCE_NAME already exists"
    if [ "$FORCE" = true ]; then
        echo "🛑 Stopping and removing existing container: $INSTANCE_NAME"
        docker stop "$INSTANCE_NAME" >/dev/null 2>&1
        docker rm "$INSTANCE_NAME" >/dev/null 2>&1
        echo "✅ Container removed"
    else
        echo "❌ Container already exists. Use --force to replace it."
        exit 1
    fi
fi

# Prepare Docker run command
DOCKER_ARGS=(run -d)

if [ "$PULL_ALWAYS" = true ]; then
  DOCKER_ARGS+=(--pull always)
fi

DOCKER_ARGS+=(--name "$INSTANCE_NAME" -p ${PORT}:${PORT})

# Set environment mode
if [ "$DEVELOPMENT" = true ]; then
    DOCKER_ARGS+=(-e FLOUDS_API_ENV=Development -e FLOUDS_DEBUG_MODE=1)
    echo "Running in Development mode"
else
    DOCKER_ARGS+=(-e FLOUDS_API_ENV=Production -e FLOUDS_DEBUG_MODE=0)
    echo "Running in Production mode"
fi

# Add config file mapping
docker_config_path="${FLOUDS_ONNX_CONFIG_FILE:-/flouds-ai/app/config/onnx_config.json}"
echo "Mapping ONNX config: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST → $docker_config_path"
DOCKER_ARGS+=(-v "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST:$docker_config_path:ro" -e "FLOUDS_ONNX_CONFIG_FILE=$docker_config_path")

# Add ONNX model directory mapping if exists
if [[ -n "$FLOUDS_ONNX_MODEL_PATH_AT_HOST" ]]; then
    docker_onnx_path="${FLOUDS_ONNX_ROOT:-/flouds-ai/onnx}"
    echo "Mapping ONNX models: $FLOUDS_ONNX_MODEL_PATH_AT_HOST → $docker_onnx_path"
    DOCKER_ARGS+=(-v "$FLOUDS_ONNX_MODEL_PATH_AT_HOST:$docker_onnx_path:ro" -e "FLOUDS_ONNX_ROOT=$docker_onnx_path")
fi

# Log directory mapping
if [[ -n "$FLOUDS_LOG_PATH_AT_HOST" ]]; then
    docker_log_path="/flouds-ai/logs"
    echo "Mapping logs: $FLOUDS_LOG_PATH_AT_HOST → $docker_log_path"
    DOCKER_ARGS+=(-v "$FLOUDS_LOG_PATH_AT_HOST:$docker_log_path:rw" -e "FLOUDS_LOG_PATH=$docker_log_path")
fi

# Add platform flag if specified
if [[ -n "$DOCKER_PLATFORM" ]]; then
    echo "Setting platform: $DOCKER_PLATFORM"
    DOCKER_ARGS+=(--platform "$DOCKER_PLATFORM")
fi

# Add image name
DOCKER_ARGS+=("$FULL_IMAGE_NAME")

echo
echo "== Starting Flouds.AI container =="
echo "Command: docker ${DOCKER_ARGS[*]}"
if docker "${DOCKER_ARGS[@]}"; then
    echo "✅ Flouds.AI container started successfully"
    
    # Wait for container to initialize
    echo "Waiting for container to initialize..."
    sleep 5
    
    echo "========================================================="
    echo "Container Status:"
    docker ps --filter "name=$INSTANCE_NAME" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    echo "========================================================="
    echo "API available at: http://localhost:${PORT}/docs"
else
    echo "❌ Failed to start Flouds.AI container."
    exit 1
fi

echo "========================================================="
echo "Container Management:"
echo "  * View logs: docker logs -f $INSTANCE_NAME"
echo "  * Stop container: docker stop $INSTANCE_NAME"
echo "  * Remove container: docker rm $INSTANCE_NAME"
echo

read -p "Would you like to view container logs now? (y/n) " showLogs
if [[ "$showLogs" == "y" || "$showLogs" == "Y" ]]; then
    docker logs -f "$INSTANCE_NAME"
fi
