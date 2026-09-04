"""``fairbench``: the CI entry point Chapter 9 calls, layered over ``fairbench_genai``.

The installable library in this repository is called ``fairbench_genai``; the
name ``fairbench`` was already taken on PyPI. Chapter 9 prints a CI workflow
whose benchmark step runs::

    python -m fairbench.run --trigger ... --scenario ... --model-version ...

so this package exists to make that command real. It is a thin shim, and it is
worth being precise about what "thin" means here, because a silent alias would
be worse than no shim at all:

- ``fairbench`` is **not** a copy of the library and **not** an import alias for
  it. It contains no metric code, no adapters, and no scenario handling. Every
  number it reports is computed by ``fairbench_genai``.
- It adds the layer the chapter is about and the library does not have: trigger
  validation, model-version attribution, gate evaluation, and a process exit
  code a CI job can act on. That layer is genuinely a deployment concern rather
  than a measurement one, which is why it sits outside the library.
- It lives at the repository root instead of under ``src/``, so it is not part
  of the ``fairbench-genai`` distribution. Installing the library gives you the
  library; checking out this repository gives you the workflow around it. A CI
  job that has run ``actions/checkout`` can call ``python -m fairbench.run``
  from the checkout directory, which is exactly what the printed workflow does.

If you want the library's own interface, use ``fairbench_genai`` (or its
``fairbench`` console script, installed from ``pyproject.toml``) directly.
"""

from __future__ import annotations

__all__ = ["__version__", "LIBRARY_PACKAGE"]

# The package this shim delegates all measurement to.
LIBRARY_PACKAGE = "fairbench_genai"

__version__ = "0.1.0"
