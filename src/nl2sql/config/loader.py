"""Layered configuration loading.

Sources, each overriding the previous one:

1. ``configs/base.yaml``: defaults that are not environment specific
2. ``configs/<env>.yaml``: the differences for one deployment
3. ``NL2SQL_`` environment variables, including values from a local ``.env``

Mappings merge recursively. Lists and scalars are replaced outright, so an
environment file can remove a default list entry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError as PydanticValidationError

from nl2sql.config.settings import Settings
from nl2sql.core.exceptions import ConfigurationError

ENV_VAR = "NL2SQL_ENV"
CONFIG_DIR_VAR = "NL2SQL_CONFIG_DIR"
DEFAULT_CONFIG_DIR = Path("configs")
BASE_FILENAME = "base.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``override`` merged onto ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration file {path}.") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Configuration file {path} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration file {path} must contain a mapping.")
    return data


def resolve_environment(env: str | None = None) -> str:
    """Return the active environment name."""
    return (env or os.getenv(ENV_VAR) or "development").strip().lower()


def resolve_config_dir(config_dir: Path | str | None = None) -> Path:
    """Return the directory holding the YAML layers."""
    if config_dir is not None:
        return Path(config_dir)
    from_env = os.getenv(CONFIG_DIR_VAR)
    return Path(from_env) if from_env else DEFAULT_CONFIG_DIR


def load_config_mapping(
    *, env: str | None = None, config_dir: Path | str | None = None
) -> dict[str, Any]:
    """Return the merged YAML layers without validating them."""
    directory = resolve_config_dir(config_dir)
    environment = resolve_environment(env)
    base_path = directory / BASE_FILENAME
    env_path = directory / f"{environment}.yaml"

    if not base_path.is_file():
        raise ConfigurationError(f"Base configuration {base_path} was not found.")

    merged = _read_yaml(base_path)
    if env_path.is_file():
        merged = deep_merge(merged, _read_yaml(env_path))
    else:
        raise ConfigurationError(
            f"Environment configuration {env_path} was not found for environment {environment}."
        )
    merged["env"] = environment
    merged["config_dir"] = str(directory)
    return merged


def load_settings(
    *,
    env: str | None = None,
    config_dir: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    load_dotenv_file: bool = True,
) -> Settings:
    """Build and validate the settings object.

    Args:
        env: Environment name. Defaults to ``NL2SQL_ENV`` then ``development``.
        config_dir: Directory with the YAML layers.
        overrides: Values merged last, used by tests and the CLI.
        load_dotenv_file: Load ``.env`` into the process environment first,
            without replacing variables that are already set.
    """
    if load_dotenv_file:
        load_dotenv(override=False)
    mapping = load_config_mapping(env=env, config_dir=config_dir)
    if overrides:
        mapping = deep_merge(mapping, overrides)
    try:
        return Settings(**mapping)
    except PydanticValidationError as exc:
        lines = ["Configuration validation failed:"]
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "<root>"
            lines.append(f"  - {location}: {error['msg']}")
        raise ConfigurationError("\n".join(lines)) from exc
