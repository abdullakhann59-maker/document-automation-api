class AppError(Exception):
    """An error with an HTTP status; `extra` is merged into the error response."""

    def __init__(self, status_code: int, message: str, **extra):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.extra = extra
