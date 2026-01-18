#!/bin/bash
# =============================================================================
# File: start-flouds-ai.sh
# Date: 2024-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
#
# This script sets up and runs the Flouds AI container with proper volume mapping
# and environment variable handling based on a .env file.
#
# Usage:
#   ./start-flouds-ai.sh [--env-file <path>] [--instance <name>] [--image <name>] [--tag <tag>] [--force] [--build] [--pull-always] [--gpu] [--development]
#
# Parameters:
#   --env-file       : Path to .env file (default: ".env")
#   --instance       : Name of the Docker container (default: "flouds-ai-instance")
#   --image          : Base Docker image name (default: "gmalakar/flouds-ai-cpu")
#   --tag            : Tag for the Docker image (default: "latest")
#   --gpu            : Use GPU image (default: false)
#   --force          : Force restart container if it exists
#   --build          : Build Docker image locally before starting container
#   --pull-always    : Always pull image from registry before running
#   --development    : Run in development mode
# =============================================================================

# Default values
ENV_FILE=".env"
INSTANCE_NAME="flouds-ai-instance"
IMAGE_NAME="gmalakar/flouds-ai-cpu"
TAG="latest"
GPU=false
FORCE=false
BUILD_IMAGE=false
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
            GPU=true
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
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions
write_step_header() {
    echo -e "\n== $1 =="
}

write_success() {
    echo -e " \033[32m$1\033[0m"
}

write_warning() {
    echo -e " \033[33m$1\033[0m"
}

write_error() {
    echo -e " \033[31m$1\033[0m"
    exit 1
}

test_docker() {
    if ! command -v docker &> /dev/null; then
        write_error "Docker is not installed or not in PATH"
    fi

    if ! docker version &> /dev/null; then
        write_error "Docker is not running or not accessible"
    fi

    write_success "Docker is running"
}

read_env_file() {
    local file_path="$1"
    if [[ ! -f "$file_path" ]]; then
        return 1
    fi

    # Export variables from .env file
    set -a
    source "$file_path"
    set +a
}

set_directory_permissions() {
    local path="$1"
    local description="$2"

    if [[ ! -d "$path" ]]; then
        write_warning "$description directory does not exist: $path"
        echo "Creating directory..."
        if ! mkdir -p "$path"; then
            write_error "Failed to create $description directory: $path"
        fi
        write_success "$description directory created: $path"
    else
        write_success "Found $description directory: $path"
    fi

    # Test if directory is writable
    if [[ ! -w "$path" ]]; then
        write_warning "$description directory is not writable: $path"
        echo "Setting permissions on $description directory..."
        if ! chmod 755 "$path"; then
            write_warning "Failed to set permissions: $path"
            read -p "Continue anyway? (y/n): " continue_choice
            if [[ "$continue_choice" != "y" ]]; then
                echo "Aborted by user."
                exit 0
            fi
        else
            write_success "Permissions set successfully"
        fi
    else
        write_success "$description directory is writable: $path"
    fi
}

# Adjust image name for GPU
if [[ "$GPU" == true && "$IMAGE_NAME" == "gmalakar/flouds-ai-cpu" ]]; then
    IMAGE_NAME="gmalakar/flouds-ai-gpu"
fi
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

WORKING_DIR="/flouds-ai"

echo "========================================================="
echo "                 Flouds AI STARTER SCRIPT                "
echo "========================================================="
echo "Instance Name : $INSTANCE_NAME"
echo "Base Image    : $IMAGE_NAME"
echo "Tag           : $TAG"
echo "Full Image    : $FULL_IMAGE_NAME"
echo "Environment   : $ENV_FILE"
echo "Build Image   : $BUILD_IMAGE"
echo "Force Restart : $FORCE"
echo "Pull Always   : $PULL_ALWAYS"
echo "GPU Mode      : $GPU"
echo "Development   : $DEVELOPMENT"
echo "========================================================="

write_step_header "Checking Docker installation"
test_docker

write_step_header "Reading environment configuration"
if [[ ! -f "$ENV_FILE" ]]; then
    write_error "$ENV_FILE not found. Please create this file with required environment variables."
fi
write_success "Using environment file: $ENV_FILE"

if ! read_env_file "$ENV_FILE"; then
    write_error "Failed to read environment file: $ENV_FILE"
fi

write_step_header "Validating required environment variables"
if [[ -z "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST" ]]; then
    write_error "FLOUDS_ONNX_CONFIG_FILE_AT_HOST environment variable is required but not set in $ENV_FILE"
fi

if [[ ! -f "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST" ]]; then
    write_error "ONNX config file not found: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST"
fi
write_success "Found ONNX config file: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST"

if [[ -z "$FLOUDS_ONNX_MODEL_PATH_AT_HOST" ]]; then
    write_error "FLOUDS_ONNX_MODEL_PATH_AT_HOST environment variable is required but not set in $ENV_FILE"
fi

if [[ ! -d "$FLOUDS_ONNX_MODEL_PATH_AT_HOST" ]]; then
    write_error "ONNX model path does not exist: $FLOUDS_ONNX_MODEL_PATH_AT_HOST"
fi
write_success "Found ONNX model path: $FLOUDS_ONNX_MODEL_PATH_AT_HOST"

# Check and set permissions for log directory
if [[ -n "$FLOUDS_LOG_PATH_AT_HOST" ]]; then
    set_directory_permissions "$FLOUDS_LOG_PATH_AT_HOST" "Log"
else
    write_warning "FLOUDS_LOG_PATH_AT_HOST not set. Container logs will not be persisted to host."
fi

# Check and set permissions for Data directory (SQLite database)
if [[ -n "$FLOUDS_DATA_PATH_AT_HOST" ]]; then
    set_directory_permissions "$FLOUDS_DATA_PATH_AT_HOST" "Data"
else
    write_warning "FLOUDS_DATA_PATH_AT_HOST not set. Client database will not be persisted to host."
fi

if [[ "$BUILD_IMAGE" == true ]]; then
    write_step_header "Building Docker image"
    echo "Building $FULL_IMAGE_NAME..."
    if ! docker build -t "$FULL_IMAGE_NAME" .; then
        write_error "Failed to build Docker image."
    fi
    write_success "Docker image built successfully: $FULL_IMAGE_NAME"
fi

write_step_header "Setting up Docker network"
AI_NETWORK="flouds_ai_network"
if ! docker network ls --format '{{.Name}}' | grep -q "^${AI_NETWORK}$"; then
    echo "Creating network: $AI_NETWORK"
    if ! docker network create "$AI_NETWORK" > /dev/null; then
        write_error "Failed to create network: $AI_NETWORK"
    fi
    write_success "Network $AI_NETWORK created successfully"
else
    write_success "Network $AI_NETWORK already exists"
fi

write_step_header "Managing container instance"
if docker ps -a --format '{{.Names}}' | grep -q "^${INSTANCE_NAME}$"; then
    write_warning "Container $INSTANCE_NAME already exists"
    if [[ "$FORCE" == true ]]; then
        echo "Stopping and removing existing container: $INSTANCE_NAME"
        docker stop "$INSTANCE_NAME" > /dev/null 2>&1
        docker rm "$INSTANCE_NAME" > /dev/null 2>&1
        write_success "Container removed"
    else
        write_error "Container already exists. Use --force to replace it."
    fi
fi

write_step_header "Preparing container configuration"
DOCKER_ARGS=("run" "-d")

if [[ "$PULL_ALWAYS" == true ]]; then
    DOCKER_ARGS+=("--pull" "always")
fi

DOCKER_ARGS+=(
    "--name" "$INSTANCE_NAME"
    "--network" "$AI_NETWORK"
    "-p" "19690:19690"
)

# Cache-related default environment variables (applied only if missing in .env)
declare -A CACHE_DEFAULTS=(
    [FLOUDS_ENCODER_CACHE_MAX]="3"
    [FLOUDS_DECODER_CACHE_MAX]="3"
    [FLOUDS_MODEL_CACHE_MAX]="2"
    [FLOUDS_SPECIAL_TOKENS_CACHE_MAX]="8"
    [FLOUDS_GENERATION_CACHE_MAX]="256"
    [FLOUDS_ENCODER_OUTPUT_CACHE_MAX]="128"
    [FLOUDS_ENCODER_OUTPUT_CACHE_MAX_BYTES]="10485760" # 10MB
)

# Load all environment variables from .env file
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ $key =~ ^#.*$ ]] && continue
    [[ -z $key ]] && continue

    # Skip _AT_HOST variables as they are for host path mapping
    [[ $key =~ .*_AT_HOST$ ]] && continue

    # Remove quotes from value
    value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")

    DOCKER_ARGS+=("-e" "$key=$value")
done < "$ENV_FILE"

# Inject cache defaults only if not set via .env
for key in "${!CACHE_DEFAULTS[@]}"; do
    if [[ -z "${!key}" ]]; then
        value="${CACHE_DEFAULTS[$key]}"
        DOCKER_ARGS+=("-e" "$key=$value")
        echo "Setting default $key=$value"
    fi
done

# Override environment based on development flag
if [[ "$DEVELOPMENT" == true ]]; then
    DOCKER_ARGS+=("-e" "FLOUDS_API_ENV=Development")
    DOCKER_ARGS+=("-e" "APP_DEBUG_MODE=1")
fi

# ONNX config file mapping
if [[ -n "$FLOUDS_ONNX_CONFIG_FILE_AT_HOST" ]]; then
    DOCKER_CONFIG_PATH="$WORKING_DIR/app/config/onnx_config.json"
    echo "Mapping ONNX config: $FLOUDS_ONNX_CONFIG_FILE_AT_HOST -> $DOCKER_CONFIG_PATH"
    DOCKER_ARGS+=("-v" "${FLOUDS_ONNX_CONFIG_FILE_AT_HOST}:${DOCKER_CONFIG_PATH}:ro")
    DOCKER_ARGS+=("-e" "FLOUDS_ONNX_CONFIG_FILE=$DOCKER_CONFIG_PATH")
fi

# ONNX model directory mapping
if [[ -n "$FLOUDS_ONNX_MODEL_PATH_AT_HOST" ]]; then
    DOCKER_ONNX_PATH="$WORKING_DIR/onnx"
    echo "Mapping ONNX models: $FLOUDS_ONNX_MODEL_PATH_AT_HOST -> $DOCKER_ONNX_PATH"
    DOCKER_ARGS+=("-v" "${FLOUDS_ONNX_MODEL_PATH_AT_HOST}:${DOCKER_ONNX_PATH}:ro")
    DOCKER_ARGS+=("-e" "FLOUDS_ONNX_ROOT=$DOCKER_ONNX_PATH")
fi

# Log directory mapping
if [[ -n "$FLOUDS_LOG_PATH_AT_HOST" ]]; then
    DOCKER_LOG_PATH="$WORKING_DIR/logs"
    echo "Mapping logs: $FLOUDS_LOG_PATH_AT_HOST -> $DOCKER_LOG_PATH"
    DOCKER_ARGS+=("-v" "${FLOUDS_LOG_PATH_AT_HOST}:${DOCKER_LOG_PATH}:rw")
    DOCKER_ARGS+=("-e" "FLOUDS_LOG_PATH=$DOCKER_LOG_PATH")
fi

# Data directory mapping (SQLite database)
if [[ -n "$FLOUDS_DATA_PATH_AT_HOST" ]]; then
    DOCKER_DATA_PATH="$WORKING_DIR/data"
    echo "Mapping Data directory: $FLOUDS_DATA_PATH_AT_HOST -> $DOCKER_DATA_PATH"
    DOCKER_ARGS+=("-v" "${FLOUDS_DATA_PATH_AT_HOST}:${DOCKER_DATA_PATH}:rw")
    DOCKER_ARGS+=("-e" "FLOUDS_CLIENTS_DB=$DOCKER_DATA_PATH/clients.db")
fi

# Add platform flag if specified
if [[ -n "$DOCKER_PLATFORM" ]]; then
    echo "Setting platform: $DOCKER_PLATFORM"
    DOCKER_ARGS+=("--platform" "$DOCKER_PLATFORM")
fi

DOCKER_ARGS+=("$FULL_IMAGE_NAME")

write_step_header "Starting Flouds AI container"
echo "Command: docker ${DOCKER_ARGS[*]}"

if docker "${DOCKER_ARGS[@]}"; then
    write_success "Flouds AI container started successfully"
    write_success "API available at: http://localhost:19690/docs"
    echo "Waiting for container to initialize..."
    sleep 3

    write_step_header "Container Status"
    docker ps --filter "name=$INSTANCE_NAME" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
else
    write_error "Failed to start Flouds AI container."
fi

write_step_header "Container Management"
echo "Use the following commands to manage the container:"
echo "  * View logs: docker logs -f $INSTANCE_NAME"
echo "  * Stop container: docker stop $INSTANCE_NAME"
echo "  * Remove container: docker rm $INSTANCE_NAME"
echo "  * View API docs: http://localhost:19690/docs"
echo ""

read -p "Would you like to view container logs now? (y/n): " show_logs
if [[ "$show_logs" == "y" || "$show_logs" == "Y" ]]; then
    docker logs -f "$INSTANCE_NAME"
fi
