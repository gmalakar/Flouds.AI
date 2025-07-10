#!/bin/bash

set -e

ENV_FILE=".env"
INSTANCE_NAME="flouds-ai-instance"
IMAGE_NAME="gmalakar/flouds-ai-cpu:latest"
AI_NETWORK="flouds_ai_network"
PORT=19690

echo "========================================================="
echo "                 FLOUDS.AI STARTER SCRIPT                "
echo "========================================================="
echo "Instance Name : $INSTANCE_NAME"
echo "Image         : $IMAGE_NAME"
echo "Environment   : $ENV_FILE"
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

# Read .env file
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ $ENV_FILE not found. Please create this file with required environment variables."
    exit 1
fi
echo "✅ Using environment file: $ENV_FILE"
set -o allexport
source "$ENV_FILE"
set +o allexport

# Validate required environment variables
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

# Ensure Docker network exists
ensure_network "$AI_NETWORK"

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${INSTANCE_NAME}$"; then
    echo "🛑 Stopping and removing existing container: $INSTANCE_NAME"
    docker stop "$INSTANCE_NAME"
    docker rm "$INSTANCE_NAME"
    echo "✅ Container removed"
fi

# Prepare Docker run command
DOCKER_ARGS=(run -d --name "$INSTANCE_NAME" --network "$AI_NETWORK" -p ${PORT}:${PORT} -e FLOUDS_API_ENV=Production -e FLOUDS_DEBUG_MODE=0)

# ONNX config file mapping
docker_config_path="${FLOUDS_ONNX_CONFIG_FILE:-/flouds-ai/app/config/onnx_config.json}"
echo "Mapping ONNX config: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST → $docker_config_path"
DOCKER_ARGS+=(-v "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST:$docker_config_path:ro" -e "FLOUDS_ONNX_CONFIG_FILE=$docker_config_path")

# ONNX model directory mapping
if [[ -n "$FLOUDS_ONNX_MODEL_PATH_AT_HOST" ]]; then
    docker_onnx_path="${FLOUDS_ONNX_ROOT:-/flouds-ai/onnx}"
    echo "Mapping ONNX models: $FLOUDS_ONNX_MODEL_PATH_AT_HOST → $docker_onnx_path"
    DOCKER_ARGS+=(-v "$FLOUDS_ONNX_MODEL_PATH_AT_HOST:$docker_onnx_path:ro" -e "FLOUDS_ONNX_ROOT=$docker_onnx_path")
fi

# Log directory mapping
if [[ -n "$FLOUDS_LOG_PATH_AT_HOST" ]]; then
    docker_log_path="${FLOUDS_LOG_PATH:-/var/log/flouds-ai}"
    echo "Mapping logs: $FLOUDS_LOG_PATH_AT_HOST → $docker_log_path"
    DOCKER_ARGS+=(-v "$FLOUDS_LOG_PATH_AT_HOST:$docker_log_path:rw" -e "FLOUDS_LOG_PATH=$docker_log_path")
fi

DOCKER_ARGS+=("$IMAGE_NAME")

echo "========================================================="
echo "Starting Flouds.AI container..."
echo "Command: docker ${DOCKER_ARGS[*]}"
if docker "${DOCKER_ARGS[@]}"; then
    echo "✅ Flouds.AI container started successfully"
    echo "API available at: http://localhost:${PORT}/docs"
    echo "Waiting for container to initialize..."
    sleep 3
    echo "========================================================="
    echo "Container Status:"
    docker ps --filter "name=$INSTANCE_NAME" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
else
    echo "❌ Failed to start Flouds.Py container."
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