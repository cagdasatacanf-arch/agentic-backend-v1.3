"""
Secrets Loader - Preloads secrets into environment before config initialization

This module MUST be imported before app.config to ensure secrets are loaded
from Docker Secrets or secret files before Pydantic Settings reads them.

Usage:
    # In app/main.py or app/__init__.py (BEFORE importing config):
    from app.utils import secrets_loader  # noqa - side effect import
    from app.config import settings  # Now reads from populated environment
"""

import os
import logging
from pathlib import Path

# Configure logging for secrets loader
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_secrets_to_environment():
    """
    Load secrets from Docker Secrets or secret files into environment variables.

    This allows Pydantic Settings to seamlessly pick up secrets without modification.

    Priority order:
    1. Existing environment variable (don't override)
    2. Docker Secret at /run/secrets/<lowercase_name>
    3. File path from <NAME>_FILE environment variable
    """

    # Define which environment variables should be loaded from secrets
    SECRET_VARS = [
        "OPENAI_API_KEY",
        "LANGCHAIN_API_KEY",
        "REDIS_PASSWORD",
        "QDRANT_API_KEY",
        "INTERNAL_API_KEY",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
    ]

    secrets_loaded = 0
    secrets_skipped = 0

    for env_var in SECRET_VARS:
        # Skip if already set in environment
        if os.getenv(env_var):
            logger.debug(f"Secret {env_var} already in environment, skipping")
            secrets_skipped += 1
            continue

        # Try Docker Secret file
        docker_secret_path = Path(f"/run/secrets/{env_var.lower()}")
        if docker_secret_path.exists():
            try:
                value = docker_secret_path.read_text().strip()
                os.environ[env_var] = value
                logger.info(f"✓ Loaded {env_var} from Docker secret")
                secrets_loaded += 1
                continue
            except Exception as e:
                logger.warning(f"Failed to read Docker secret {env_var}: {e}")

        # Try file path from <ENV_VAR>_FILE
        file_path_env = f"{env_var}_FILE"
        file_path = os.getenv(file_path_env)
        if file_path:
            try:
                value = Path(file_path).read_text().strip()
                os.environ[env_var] = value
                logger.info(f"✓ Loaded {env_var} from file: {file_path}")
                secrets_loaded += 1
                continue
            except Exception as e:
                logger.warning(f"Failed to read secret file {file_path}: {e}")

        # Secret not found - log debug (not error, as some secrets are optional)
        logger.debug(f"Secret {env_var} not found in any source")

    if secrets_loaded > 0:
        logger.info(f"Secrets loader: {secrets_loaded} secrets loaded, {secrets_skipped} already set")


# AUTO-RUN: Load secrets when this module is imported
# This allows simple "import secrets_loader" to trigger loading
load_secrets_to_environment()
