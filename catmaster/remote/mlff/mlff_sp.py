try:  # Package import in tests; flat import after DPDispatcher staging.
    from .mlff_common import cli
except ImportError:  # pragma: no cover - exercised by remote staged scripts
    from mlff_common import cli


if __name__ == "__main__":
    cli("sp")
