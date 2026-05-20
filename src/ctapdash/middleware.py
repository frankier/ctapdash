from contextvars import ContextVar
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware


_request_var: ContextVar[Request | None] = ContextVar("request_var", default=None)


def get_current_request() -> Request | None:
    """Retrieve the current request globally."""
    return _request_var.get()


class GlobalRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Set the request in the context at the start of the lifecycle
        token = _request_var.set(request)
        try:
            response = await call_next(request)
        finally:
            # Reset context var to avoid memory leaks
            _request_var.reset(token)
        return response
