import importlib
import os
import subprocess

from app.config.config_loader import ConfigLoader
from app.logger import get_logger

logger = get_logger("setup")

# Load settings using AppSettingsLoader
APP_SETTINGS = ConfigLoader.get_app_settings()

logger.info(f"Environment: {os.getenv('FASTAPI_ENV', 'Production')}")


# Install the required server dynamically
def install_server(server_name):
    """Install the required ASGI server dynamically."""
    try:
        subprocess.run(["pip", "install", server_name], check=True)
        logger.info(f"{server_name} installed successfully.")
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {server_name}.")


# Dynamically import the ASGI server
def dynamic_import(module_name):
    """Dynamically import a module using a variable."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        logger.error(f"Module {module_name} not found.")
        return None


install_server(APP_SETTINGS.server.type.lower())
SERVER_MODULE = dynamic_import(APP_SETTINGS.server.type.lower())
