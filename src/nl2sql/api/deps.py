"""Request dependencies.

Authentication, rate limiting and access to the container. The principal
produced here is the only source of a tenant identifier anywhere in the
system, which is what makes tenant isolation hold: there is no code path that
takes a tenant from a body, a query string or a header the caller controls.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from nl2sql.container import Container
from nl2sql.core.exceptions import AuthorizationError, RateLimitExceededError
from nl2sql.security.auth import ROLE_ADMIN, Principal
from nl2sql.security.rate_limit import RateLimiter


def get_container(request: Request) -> Container:
    """Return the container built at startup."""
    container: Container = request.app.state.container
    return container


def get_rate_limiter(request: Request) -> RateLimiter:
    """Return the process wide rate limiter."""
    limiter: RateLimiter = request.app.state.rate_limiter
    return limiter


def get_principal(
    request: Request,
    container: Annotated[Container, Depends(get_container)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> Principal:
    """Authenticate the caller and count the request against their limit."""
    authenticator = container.authenticator
    presented = request.headers.get(authenticator.header_name)
    principal = authenticator.authenticate(presented)
    if not limiter.check(principal.id):
        raise RateLimitExceededError("The request rate limit for this principal has been exceeded.")
    return principal


def require_admin(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    """Require the admin role, used for operations that change server state."""
    if not principal.has_role(ROLE_ADMIN):
        raise AuthorizationError("This operation requires the admin role.")
    return principal


ContainerDep = Annotated[Container, Depends(get_container)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]
AdminDep = Annotated[Principal, Depends(require_admin)]
