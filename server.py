from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unmask.api.app import create_app
from unmask.cli import main

app = create_app()

if __name__ == "__main__":
    main()
