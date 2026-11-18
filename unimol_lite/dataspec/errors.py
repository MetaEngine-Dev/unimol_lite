"""Exception types of the SMS-DataSpec adapter."""


class DataSpecError(Exception):
    """Container I/O or protocol-level failure."""


class DataSpecValidationError(DataSpecError):
    """A container (or envelope) violates the protocol; it must not be used."""


class DataSpecMismatchError(DataSpecError):
    """Two containers disagree on contract fields (fingerprint); mixing is refused."""
