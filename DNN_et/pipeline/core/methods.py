from enum import Enum


class FFMethod(str, Enum):
    """Supported final fake-factor construction methods."""

    UNCORRECTED = "none"
    NONCLOSURE = "nonclosure"
    DRSR = "drsr"
    DRSR_NONCLOSURE = "drsr_nonclosure"

    @classmethod
    def choices(cls):
        return tuple(method.value for method in cls)

    @classmethod
    def parse(cls, value):
        return value if isinstance(value, cls) else cls(value)

    @property
    def uses_drsr(self):
        return self in (self.DRSR, self.DRSR_NONCLOSURE)

    @property
    def uses_nonclosure(self):
        return self in (self.NONCLOSURE, self.DRSR_NONCLOSURE)
