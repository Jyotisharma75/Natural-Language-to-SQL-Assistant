"""Secret resolution.

Settings name the environment variable that holds a secret. Resolution happens
at the point of use, so the value never sits on the settings object. In Azure,
the variables are populated from Key Vault references on the container app, so
this module needs no vault client of its own.
"""

from __future__ import annotations

import os

from nl2sql.core.exceptions import SecretNotFoundError


def resolve_secret(name: str | None) -> str | None:
    """Return the secret stored in the environment variable ``name``, if any."""
    if not name:
        return None
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def require_secret(name: str | None, *, purpose: str) -> str:
    """Return the secret in ``name`` or raise a configuration error naming the variable."""
    value = resolve_secret(name)
    if value is None:
        raise SecretNotFoundError(
            f"The secret for {purpose} was not found. Set the environment variable {name}."
        )
    return value
