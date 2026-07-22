from __future__ import annotations

import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rect_detector.cli.pipeline_product import main


if __name__ == "__main__":
    main()
