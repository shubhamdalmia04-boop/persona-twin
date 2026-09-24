"""Entry point of PersonaTwin.exe (also runs from source: python launcher/main.py)."""
from __future__ import annotations

import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        raise SystemExit("Tkinter is missing. Install it (Linux: sudo apt install python3-tk) and try again.")
    from launcher import core
    from launcher.gui import App

    core.sync_app()
    App().mainloop()


if __name__ == "__main__":
    main()
