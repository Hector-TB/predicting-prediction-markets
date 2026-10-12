"""FastAPI backend: search a Polymarket market and predict it with the production release (ADR-017, ADR-025).

    uvicorn api.main:app --reload        # from the repo root
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The pipeline, data and release modules are scripts that import their siblings by
# bare name; the API reuses them as they are.
for sub in ("data_collection_pipeline", "data", "scripts", ""):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
