"""Adapters for training frameworks.

Each adapter lives in its own module and imports its framework when that
module is imported, so ``import nuviz`` never pulls in a framework:

* :mod:`nuviz.integrations.lightning` — ``NuvizLogger`` for Lightning
* :mod:`nuviz.integrations.transformers` — ``NuvizCallback`` for the
  Hugging Face ``Trainer``
"""
