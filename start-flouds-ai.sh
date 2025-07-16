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
# =============================================================================

set -e

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
echo "========================================================="

# Helper: ensure network exists
ensure_network() {
    local name="$1"
    if ! docker network ls --format '{{.Name}}' | grep -q "^${name}$"; then
        echo "🔧 Creating network: $name"
        docker network create "$name"
        echo "✅ Network $name created successfully"
    else
        echo "✅ Network $name already exists"
    fi
}

# Helper: attach network if not connected
attach_network_if_not_connected() {
    local container="$1"
    local network="$2"
    
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "⚠️ Container $container is not running. Skipping network attachment."
        return
    fi
    
    # Check if container is already connected to the network
    if ! docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$container" | grep -q "$network"; then
        echo "🔗 Attaching network $network to container $container"
        if docker network connect "$network" "$container" 2>/dev/null; then
            echo "✅ Successfully connected $container to $network"
        else
            echo "⚠️ Failed to connect $container to $network"
        fi
    else
        echo "✅ Container $container is already connected to $network"
    fi
}

# Read .env file
echo
echo "== Reading environment configuration =="
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ $ENV_FILE not found. Please create this file with required environment variables."
    exit 1
fi
echo "✅ Using environment file: $ENV_FILE"
set -o allexport
source "$ENV_FILE"
set +o allexport

# Validate required environment variables
echo
echo "== Validating required environment variables =="
if [[ -z "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST" ]]; then
    echo "❌ FLOUDS_ONNX_CONFIG_FILE_AT_HOST environment variable is required but not set in $ENV_FILE"
    exit 1
fi
if [[ ! -f "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST" ]]; then
    echo "❌ ONNX config file not found: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST"
    echo "This file is required. Please check the path and try again."
    exit 1
fi
echo "✅ Found ONNX config file: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST"

if [[ -z "$FLOUDS_ONNX_MODEL_PATH_AT_HOST" ]]; then
    echo "⚠️ FLOUDS_ONNX_MODEL_PATH_AT_HOST not set. Container will use internal models only."
else
    if [[ ! -d "$FLOUDS_ONNX_MODEL_PATH_AT_HOST" ]]; then
        echo "⚠️ ONNX model path does not exist: $FLOUDS_ONNX_MODEL_PATH_AT_HOST"
        echo "Creating directory..."
        mkdir -p "$FLOUDS_ONNX_MODEL_PATH_AT_HOST"
        echo "✅ ONNX model directory created: $FLOUDS_ONNX_MODEL_PATH_AT_HOST"
    else
        echo "✅ Found ONNX model path: $FLOUDS_ONNX_MODEL_PATH_AT_HOST"
    fi
fi

if [[ -z "$FLOUDS_LOG_PATH_AT_HOST" ]]; then
    echo "⚠️ FLOUDS_LOG_PATH_AT_HOST not set. Container logs will not be persisted to host."
else
    if [[ ! -d "$FLOUDS_LOG_PATH_AT_HOST" ]]; then
        echo "⚠️ Log directory does not exist: $FLOUDS_LOG_PATH_AT_HOST"
        echo "Creating directory..."
        mkdir -p "$FLOUDS_LOG_PATH_AT_HOST"
        echo "✅ Log directory created: $FLOUDS_LOG_PATH_AT_HOST"
    else
        echo "✅ Found log directory: $FLOUDS_LOG_PATH_AT_HOST"
    fi
fi

# Build image if requested
if [ "$BUILD_IMAGE" = true ]; then
    echo
    echo "== Building Docker image =="
    echo "Building $FULL_IMAGE_NAME..."
    
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

# Ensure Docker network exists
echo
echo "== Setting up Docker network =="
ensure_network "$AI_NETWORK"

# Stop and remove existing container if it exists
echo
echo "== Managing container instance =="
if docker ps -a --format '{{.Names}}' | grep -q "^${INSTANCE_NAME}$"; then
    echo "⚠️ Container $INSTANCE_NAME already exists"
    if [ "$FORCE" = true ]; then
        echo "🛑 Stopping and removing existing container: $INSTANCE_NAME"
        docker stop "$INSTANCE_NAME"
        docker rm "$INSTANCE_NAME"
        echo "✅ Container removed"
    else
        echo "❌ Container already exists. Use --force to replace it."
        exit 1
    fi
fi

# Prepare Docker run command
echo
echo "== Preparing container configuration =="
DOCKER_ARGS=(run -d --name "$INSTANCE_NAME" --network "$AI_NETWORK" -p ${PORT}:${PORT} -e FLOUDS_API_ENV=Production -e FLOUDS_DEBUG_MODE=0)

# ONNX config file mapping
docker_config_path="${FLOUDS_ONNX_CONFIG_FILE:-/flouds-py/app/config/onnx_config.json}"
echo "Mapping ONNX config: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST → $docker_config_path"
DOCKER_ARGS+=(-v "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST:$docker_config_path:ro" -e "FLOUDS_ONNX_CONFIG_FILE=$docker_config_path")

# ONNX model directory mapping
if [[ -n "$FLOUDS_ONNX_MODEL_PATH_AT_HOST" ]]; then
    docker_onnx_path="${FLOUDS_ONNX_ROOT:-/flouds-py/onnx}"
    echo "Mapping ONNX models: $FLOUDS_ONNX_MODEL_PATH_AT_HOST → $docker_onnx_path"
    DOCKER_ARGS+=(-v "$FLOUDS_ONNX_MODEL_PATH_AT_HOST:$docker_onnx_path:ro" -e "FLOUDS_ONNX_ROOT=$docker_onnx_path")
fi

# Log directory mapping
if [[ -n "$FLOUDS_LOG_PATH_AT_HOST" ]]; then
    docker_log_path="${FLOUDS_LOG_PATH:-/var/logs/flouds}"
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
    echo "API available at: http://localhost:${PORT}/docs"
    echo "Waiting for container to initialize..."
    sleep 3
    echo
    echo "== Container Status =="
    docker ps --filter "name=$INSTANCE_NAME" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    
    # Connect to other networks if needed
    VECTOR_NETWORK="flouds_vector_network"
    if docker network ls --format '{{.Name}}' | grep -q "^${VECTOR_NETWORK}$"; then
        echo "Connecting to vector network: $VECTOR_NETWORK"
        attach_network_if_not_connected "$INSTANCE_NAME" "$VECTOR_NETWORK"
    fi
else
    echo "❌ Failed to start Flouds.AI container."
    exit 1
fi

echo
echo "== Container Management =="
echo "Use the following commands to manage the container:"
echo "  * View logs: docker logs -f $INSTANCE_NAME"
echo "  * Stop container: docker stop $INSTANCE_NAME"
echo "  * Remove container: docker rm $INSTANCE_NAME"
echo "  * View API docs: http://localhost:${PORT}/docs"
echo

read -p "Would you like to view container logs now? (y/n) " showLogs
if [[ "$showLogs" == "y" || "$showLogs" == "Y" ]]; then
    docker logs -f "$INSTANCE_NAME"
fi