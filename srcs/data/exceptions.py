class InvalidValueError(ValueError):
    def __init__(self, msg: str = "DataFrame is NULL"):
        super().__init__(msg)
