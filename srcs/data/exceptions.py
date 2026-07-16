class InvalidValueError(ValueError):
    """Raised when a dataset value cannot be processed."""

    def __init__(self, msg: str = "DataFrame is NULL"):
        super().__init__(msg)
