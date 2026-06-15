Inference API
=============

The ``inference`` package — the §5 UQ-backed inference pipeline (Layers B and C).
It is fully decoupled from ``libuq``/``uq`` and consumes only a ``uq sample``
cache directory. See :doc:`../inference_methodology` for the theory and
:doc:`../inference_tutorial` for usage.

Cache loading
-------------

.. automodule:: inference._cache
   :members:
   :undoc-members:
   :show-inheritance:

Layer B — trajectory surrogate
------------------------------

.. automodule:: inference.parametric_dmd
   :members:
   :undoc-members:
   :show-inheritance:

Observation model (Kennedy–O'Hagan)
-----------------------------------

.. automodule:: inference.observation
   :members:
   :undoc-members:
   :show-inheritance:

Layer C — inference engines
---------------------------

.. automodule:: inference.sbi_engine
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: inference.mcmc_engine
   :members:
   :undoc-members:
   :show-inheritance:

Runnable gates
--------------

* ``inference/b1_generalization.py`` — Layer-B held-out generalization gate.
* ``inference/c_inference_prototype.py`` — Layer-C end-to-end inference +
  validation.

Both are command-line entry points; see :doc:`../inference_tutorial`.
