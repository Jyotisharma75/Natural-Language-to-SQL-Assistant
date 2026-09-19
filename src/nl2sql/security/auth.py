"""API authentication and the caller identity.

An API key is never stored, only the SHA-256 digest of it, and comparison is
constant time. The principal that comes back carries the tenant, which is the
only source the tenant scoper will accept, and the roles, which decide whether
the caller may refresh metadata.

The key material itself lives in an environment variable named by
configuration, holding a JSON array of entries::

    [{"key_sha256": "...", "principal": "reporting-app",
      "tenant_id": "acme", "roles": ["reader"]}]

A ``key`` field is accepted instead of ``key_sha256`` for local development,
and is hashed on load, so a developer does not have to run a digest by hand to
try the service.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field

from nl2sql.config.settings import ApiSettings
from nl2sql.core.exceptions import AuthenticationError, ConfigurationError

ROLE_READER = "reader"
ROLE_ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller."""

    id: str
    tenant_id: str | None = None
    roles: tuple[str, ...] = (ROLE_READER,)
    auth_method: str = "api_key"

    def has_role(self, role: str) -> bool:
        """Return whether the principal holds ``role``."""
        return role in self.roles


@dataclass(frozen=True, slots=True)
class _KeyEntry:
    digest: str
    principal: Principal


@dataclass(slots=True)
class Authenticator:
    """Resolves a presented credential into a principal."""

    settings: ApiSettings
    entries: tuple[_KeyEntry, ...] = field(default_factory=tuple)

    @property
    def header_name(self) -> str:
        """Return the header the API key is read from."""
        return self.settings.api_key.header_name

    @property
    def required(self) -> bool:
        """Return whether a credential must be presented."""
        return self.settings.auth_mode == "api_key"

    def authenticate(self, presented: str | None) -> Principal:
        """Return the principal for a presented key, or raise."""
        if self.settings.auth_mode == "none":
            return Principal(
                id="anonymous",
                tenant_id=self.settings.anonymous_tenant_id,
                roles=(ROLE_READER, ROLE_ADMIN),
                auth_method="none",
            )
        if not presented:
            raise AuthenticationError(
                f"No API key was supplied. Send it in the {self.header_name} header."
            )
        digest = hashlib.sha256(presented.encode("utf-8")).hexdigest()
        for entry in self.entries:
            if hmac.compare_digest(entry.digest, digest):
                return entry.principal
        raise AuthenticationError("The supplied API key is not recognised.")


def build_authenticator(settings: ApiSettings, raw_keys: str | None) -> Authenticator:
    """Build the authenticator from the raw contents of the keys secret."""
    if settings.auth_mode == "none":
        return Authenticator(settings=settings)
    if not raw_keys:
        raise ConfigurationError(
            "API key authentication is enabled but no keys are configured. Set the "
            f"{settings.api_key.keys_secret} environment variable to a JSON array of keys."
        )
    try:
        parsed = json.loads(raw_keys)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"The value of {settings.api_key.keys_secret} is not valid JSON."
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise ConfigurationError(
            f"{settings.api_key.keys_secret} must be a non empty JSON array of key entries."
        )

    entries: list[_KeyEntry] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ConfigurationError(f"Key entry {index} is not an object.")
        digest = item.get("key_sha256")
        if not digest and item.get("key"):
            digest = hashlib.sha256(str(item["key"]).encode("utf-8")).hexdigest()
        principal_id = item.get("principal")
        if not digest or not principal_id:
            raise ConfigurationError(
                f"Key entry {index} needs a principal and either key_sha256 or key."
            )
        roles = tuple(str(role) for role in item.get("roles") or (ROLE_READER,))
        entries.append(
            _KeyEntry(
                digest=str(digest).strip().lower(),
                principal=Principal(
                    id=str(principal_id),
                    tenant_id=(str(item["tenant_id"]) if item.get("tenant_id") else None),
                    roles=roles,
                ),
            )
        )
    return Authenticator(settings=settings, entries=tuple(entries))
