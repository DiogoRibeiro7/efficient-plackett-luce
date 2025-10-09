# Efficient Plackett-Luce

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Efficient Plackett-Luce implements fast, numerically stable learning of Plackett-Luce ranking models for data with arbitrary-sized comparison sets. The project follows the algorithmic ideas of Yeung, Kaiser, and Radicchi (2025) and provides a production-ready Python implementation with strong test coverage and documentation.

## Project Highlights

- Fast inference based on Newman's iterative updates with optional Numba acceleration
- Support for full, position-1-breaking, and projected pairwise Plackett-Luce variants
- Utilities for model evaluation, cross-validation, and benchmarking
- Examples and docs covering real-world tournament and survey use cases
- Research artifact archived under `paper/` for easy reference

## Installation

```bash
pip install efficient-plackett-luce
```

To work from source:

```bash
git clone https://github.com/diogoribeiro7/efficient-plackett-luce.git
cd efficient-plackett-luce
pip install -e .
```

## Quick Start

```python
from plackett_luce import PlackettLuceModel

# Each entry is (ranking, weight). Positions earlier in the tuple are better.
hyperedges = [
    (("TeamA", "TeamB", "TeamC"), 1),
    (("TeamB", "TeamA"), 2),
    (("TeamC", "TeamA", "TeamB"), 1),
]

model = PlackettLuceModel(model_type="full")
model.fit(hyperedges)

ranking = model.get_ranking()
probability = model.predict_probability(("TeamA", "TeamB", "TeamC"))

print("Top entities:", ranking[:3])
print("P(TeamA > TeamB > TeamC) =", probability)
```

More examples are available under `examples/`, including benchmarking scripts and cross-validation workflows.

## Repository Layout

- `plackett_luce/` – core library code and public API
- `tests/` – pytest suite covering model behavior and utilities
- `examples/` – runnable scripts demonstrating typical usage patterns
- `docs/` – Markdown documentation for API reference and tutorials
- `paper/` – research material, including [`2501.16565v1.pdf`](paper/2501.16565v1.pdf)
- `pyproject.toml` / `Makefile` – packaging configuration and development tasks

## Documentation & Learning Resources

- `docs/index.md` – project overview and conceptual background
- `docs/examples.md` – walk-throughs of end-to-end workflows
- `docs/api.md` – API reference for the exported symbols

Build the docs locally with any Markdown renderer, or integrate them into a static site generator as needed.

## Development Setup

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Handy make targets:

- `make format` – run Black and isort
- `make lint` – run static checks (flake8, mypy)
- `make test` – execute the pytest suite with coverage

## Testing

```bash
pytest
pytest --cov=plackett_luce
pytest tests/test_core.py::test_fit_convergence
```

The project maintains high coverage and exercises both synthetic and real-world style datasets.

## Research Artifact

The reference paper _Efficient inference of rankings from multi-body comparisons_ (Yeung, Kaiser, Radicchi, 2025) is bundled for convenience at `paper/2501.16565v1.pdf`. Cite it when referencing the underlying methodology.

## Citation

```bibtex
@article{yeung2025efficient,
  title={Efficient inference of rankings from multi-body comparisons},
  author={Yeung, Jack and Kaiser, Daniel and Radicchi, Filippo},
  journal={arXiv preprint arXiv:2501.16565},
  year={2025}
}
```

## Contributing

Issues and pull requests are welcome. Please open a ticket for major changes, run the test suite before submitting, and follow the code of conduct. The `CONTRIBUTING.md` file describes the full workflow.

## License

Distributed under the MIT License. See `LICENSE` for details.

## Contact

- Issues: [GitHub issue tracker](https://github.com/diogoribeiro7/efficient-plackett-luce/issues)
- Email: dfr@esmad.ipp.pt
- Paper: [arXiv:2501.16565](https://arxiv.org/abs/2501.16565)
