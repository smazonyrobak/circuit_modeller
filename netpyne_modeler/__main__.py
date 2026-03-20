from __future__ import annotations

import sys


def main() -> int:
    try:
        from .webapp import run
    except Exception as exc:
        print(
            "Unable to start the web UI. Ensure bootstrap.py created the virtual environment "
            "with the required Dash dependencies.\n"
            f"Underlying error: {exc}",
            file=sys.stderr,
        )
        return 1

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
