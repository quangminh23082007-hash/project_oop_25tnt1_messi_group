"""Existing streaming preparation and offline training commands."""


def main(argv=None):
    # Training/encoder dependencies remain lazy inside the numerical engine CLI.
    from .models._array.cli import main as engine_main
    return engine_main(argv)
