import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from snom_calculator.app import run_app

run_app()
