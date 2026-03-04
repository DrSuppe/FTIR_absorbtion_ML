"""Module entrypoint for `python -m ftir_analysis`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
