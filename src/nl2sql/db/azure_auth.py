"""Entra ID authentication for Azure SQL.

A token is passed to the ODBC driver through connection attribute 1256,
``SQL_COPT_SS_ACCESS_TOKEN``, in the packed form the driver expects: a four
byte little endian length followed by the token encoded as UTF-16LE.

Tokens are cached until shortly before they expire, so a long lived pool does
not request one per connection, and a token that expires mid process is
replaced without a restart.
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Any

from nl2sql.config.settings import DatabaseSettings
from nl2sql.core.exceptions import ConfigurationError
from nl2sql.observability.logging import get_logger

#: ODBC connection attribute that carries an access token.
SQL_COPT_SS_ACCESS_TOKEN = 1256

#: Scope requested for the Azure SQL data plane.
AZURE_SQL_SCOPE = "https://database.windows.net/.default"

#: Refresh this long before expiry, so a token cannot expire in flight.
_EXPIRY_MARGIN_SECONDS = 300

logger = get_logger(__name__)


class AzureSQLTokenProvider:
    """Supplies Entra ID access tokens for the ODBC driver."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._credential: Any | None = None
        self._token: str | None = None
        self._expires_on: float = 0.0

    def _build_credential(self) -> Any:
        try:
            from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
        except ImportError as exc:  # pragma: no cover - azure-identity is a hard dependency
            raise ConfigurationError(
                "Entra ID authentication requires the azure-identity package."
            ) from exc

        if self._settings.auth_mode == "entra_managed_identity":
            client_id = self._settings.managed_identity_client_id
            logger.info("azure_sql_credential_created", kind="managed_identity")
            return (
                ManagedIdentityCredential(client_id=client_id)
                if client_id
                else ManagedIdentityCredential()
            )
        logger.info("azure_sql_credential_created", kind="default_chain")
        return DefaultAzureCredential(
            managed_identity_client_id=self._settings.managed_identity_client_id
        )

    def get_token(self) -> str:
        """Return a valid access token, fetching or refreshing as needed."""
        with self._lock:
            now = time.time()
            if self._token and now < self._expires_on - _EXPIRY_MARGIN_SECONDS:
                return self._token
            if self._credential is None:
                self._credential = self._build_credential()
            token = self._credential.get_token(AZURE_SQL_SCOPE)
            self._token = token.token
            self._expires_on = float(token.expires_on)
            logger.info("azure_sql_token_acquired", expires_in_seconds=int(self._expires_on - now))
            return self._token

    def token_struct(self) -> bytes:
        """Return the token packed the way the ODBC driver expects it."""
        return pack_access_token(self.get_token())


def pack_access_token(token: str) -> bytes:
    """Pack an access token into the ODBC ``SQL_COPT_SS_ACCESS_TOKEN`` layout."""
    encoded = token.encode("utf-16-le")
    return struct.pack("<I", len(encoded)) + encoded
