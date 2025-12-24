"""
Secrets Management Utilities

Provides secure handling of sensitive configuration values.
Supports both Docker Secrets (production) and environment variables (development).

Usage:
    from app.utils.secrets import get_secret

    api_key = get_secret("OPENAI_API_KEY")
    # Automatically tries:
    # 1. /run/secrets/openai_api_key (Docker secret)
    # 2. OPENAI_API_KEY_FILE env var pointing to file
    # 3. OPENAI_API_KEY env var (fallback for development)
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_secret(
    env_var_name: str,
    default: Optional[str] = None,
    required: bool = False
) -> Optional[str]:
    """
    Get secret value from multiple sources in order of preference:

    1. Docker Secret file at /run/secrets/<lowercase_name>
    2. File path specified in <ENV_VAR_NAME>_FILE environment variable
    3. Direct environment variable <ENV_VAR_NAME>
    4. Default value

    Args:
        env_var_name: Name of environment variable (e.g., "OPENAI_API_KEY")
        default: Default value if secret not found
        required: If True, raises ValueError when secret not found

    Returns:
        Secret value as string

    Raises:
        ValueError: If required=True and secret not found

    Examples:
        >>> get_secret("OPENAI_API_KEY")
        'sk-...'

        >>> get_secret("OPTIONAL_KEY", default="default_value")
        'default_value'

        >>> get_secret("REQUIRED_KEY", required=True)
        ValueError: Required secret REQUIRED_KEY not found
    """

    # 1. Try Docker Secret file (/run/secrets/<lowercase_name>)
    docker_secret_path = Path(f"/run/secrets/{env_var_name.lower()}")
    if docker_secret_path.exists():
        try:
            value = docker_secret_path.read_text().strip()
            logger.debug(f"Loaded secret {env_var_name} from Docker secret")
            return value
        except Exception as e:
            logger.warning(f"Failed to read Docker secret {env_var_name}: {e}")

    # 2. Try file path from <ENV_VAR_NAME>_FILE environment variable
    file_path_env = f"{env_var_name}_FILE"
    file_path = os.getenv(file_path_env)
    if file_path:
        try:
            value = Path(file_path).read_text().strip()
            logger.debug(f"Loaded secret {env_var_name} from file: {file_path}")
            return value
        except Exception as e:
            logger.warning(f"Failed to read secret file {file_path}: {e}")

    # 3. Try direct environment variable
    value = os.getenv(env_var_name)
    if value:
        logger.debug(f"Loaded secret {env_var_name} from environment variable")
        return value

    # 4. Use default or raise error
    if required and default is None:
        raise ValueError(
            f"Required secret {env_var_name} not found. "
            f"Tried: Docker secret, {file_path_env}, {env_var_name}"
        )

    if default is not None:
        logger.debug(f"Using default value for {env_var_name}")
    else:
        logger.debug(f"Secret {env_var_name} not found, returning None")

    return default


def get_secret_int(
    env_var_name: str,
    default: Optional[int] = None,
    required: bool = False
) -> Optional[int]:
    """
    Get secret value as integer.

    Args:
        env_var_name: Name of environment variable
        default: Default value if secret not found
        required: If True, raises ValueError when secret not found

    Returns:
        Secret value as integer
    """
    value = get_secret(env_var_name, str(default) if default is not None else None, required)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Secret {env_var_name} is not a valid integer: {value}")


def get_secret_bool(
    env_var_name: str,
    default: Optional[bool] = None,
    required: bool = False
) -> Optional[bool]:
    """
    Get secret value as boolean.

    Accepts: true/false, yes/no, 1/0 (case-insensitive)

    Args:
        env_var_name: Name of environment variable
        default: Default value if secret not found
        required: If True, raises ValueError when secret not found

    Returns:
        Secret value as boolean
    """
    value = get_secret(env_var_name, str(default) if default is not None else None, required)
    if value is None:
        return None

    value_lower = value.lower().strip()
    if value_lower in ('true', 'yes', '1', 'on'):
        return True
    elif value_lower in ('false', 'no', '0', 'off'):
        return False
    else:
        raise ValueError(
            f"Secret {env_var_name} is not a valid boolean: {value}. "
            f"Use: true/false, yes/no, 1/0"
        )


def secrets_status() -> dict:
    """
    Get status of all secrets loading mechanisms.

    Useful for debugging and health checks.

    Returns:
        Dictionary with secrets status information
    """
    return {
        "docker_secrets_available": Path("/run/secrets").exists(),
        "docker_secrets_dir": str(Path("/run/secrets")),
        "docker_secrets_count": len(list(Path("/run/secrets").iterdir())) if Path("/run/secrets").exists() else 0,
        "env_vars_count": len([k for k in os.environ.keys() if not k.endswith("_FILE")]),
        "file_ref_vars_count": len([k for k in os.environ.keys() if k.endswith("_FILE")]),
    }
