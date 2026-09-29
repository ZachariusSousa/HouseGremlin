from contextvars import ContextVar


current_correlation_id: ContextVar[str | None] = ContextVar("current_correlation_id", default=None)

