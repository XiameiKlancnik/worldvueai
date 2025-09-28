from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('worldvue')
except PackageNotFoundError:
    __version__ = '0.1.0'

__all__ = ['__version__', 'get_version']


def get_version() -> str:
    """Return the current package version."""
    return __version__
