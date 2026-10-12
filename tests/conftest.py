"""Loaded by pytest before any test module: set up the import paths the API uses, so tests
can import pipeline and script modules (fetch_markets, release, …) by bare name."""

import api  # noqa: F401
