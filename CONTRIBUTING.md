# Contributing

Use Python 3.13 in a virtual environment; 3.11–3.13 are supported.
Install the pinned app from the repository root:

```bash
python -m pip install -c constraints.txt setuptools wheel
python -m pip install --no-build-isolation -c constraints.txt -e .
python -m pip check
```

Keep changes focused and follow the existing module boundaries described in
[development](docs/development.md). For scientific changes, document the equation or reference you checked against, the units and the convergence settings.

Report bugs through GitHub Issues with reproduction steps, Python/package
versions, numerical settings, and a minimal input you can share publicly.
Include expected and actual behavior. Do not upload passwords, secrets or
private experimental data. Pull requests should explain the resulting behavior
and say how you verified it. Run the tests with `python -m pip install -e ".[test]"` and `python -m pytest -q`.
