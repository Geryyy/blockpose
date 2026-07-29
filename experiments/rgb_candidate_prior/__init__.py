"""Offline RGB candidate-prior experiment (see docs/rgb-candidate-prior-plan.md).

Experimental code kept OUTSIDE ``python/blockpose/`` until it passes the
promotion criteria in ``docs/rgb-candidate-prior.md``.  Nothing here imports a
model runtime (torch) at import time; the model phases live behind the ``sam``
optional extra and are imported only when explicitly invoked.
"""

from __future__ import annotations
