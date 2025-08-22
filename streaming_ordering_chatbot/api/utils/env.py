import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


_ENV_LOADED = False


def load_env_once(env_path: Optional[Path] = None) -> None:
    """Load environment variables from a .env file once.

    If env_path is not provided, it attempts to load from the project root.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    if env_path is None:
        # Try to infer project root (3 levels up from this file)
        env_path = Path(__file__).resolve().parents[3] / ".env"

    try:
        load_dotenv(dotenv_path=env_path)
    finally:
        _ENV_LOADED = True


def get_required_env_var(name: str) -> str:
    """Get a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. Please set it in your .env file."
        )
    return value


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get an optional environment variable with a default."""
    return os.getenv(name, default)
