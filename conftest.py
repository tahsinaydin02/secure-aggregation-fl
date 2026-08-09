"""Pytest configuration.

The only job of this file is to exist at the project root: pytest prepends the
directory containing the root `conftest.py` to `sys.path`, which is what makes
`from src.secure_agg import ...` resolve inside the tests. Without it, plain
`pytest` would only work when invoked as `python -m pytest` (which adds the
current directory to the path itself).
"""
