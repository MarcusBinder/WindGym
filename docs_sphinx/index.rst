WindGym API Reference
=====================

This page provides an auto-generated reference for the main classes and functions in WindGym.

Environment Classes
-------------------

.. autoclass:: WindGym.WindFarmEnv
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.FarmEval
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.WindFarmEnvMulti
   :members:
   :undoc-members:
   :show-inheritance:

Wrappers
--------

.. autoclass:: WindGym.wrappers.CurriculumWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.wrappers.RecordEpisodeVals
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.core.NoisyWindFarmEnv
   :members:
   :undoc-members:
   :show-inheritance:

Agent Classes
-------------

.. autoclass:: WindGym.Agents.BaseAgent.BaseAgent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.Agents.PyWakeAgent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.Agents.PyWakeAgent.NoisyPyWakeAgent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.Agents.GreedyAgent.GreedyAgent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.Agents.RandomAgent.RandomAgent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.Agents.ConstantAgent.ConstantAgent
   :members:
   :undoc-members:
   :show-inheritance:

Noise Models
------------

.. autoclass:: WindGym.core.WhiteNoiseModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.core.EpisodicBiasNoiseModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.core.HybridNoiseModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindGym.core.MeasurementManager
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Tools
----------------

.. autoclass:: WindGym.utils.evaluate_PPO.Coliseum
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Layout Generation
~~~~~~~~~~~~~~~~~

.. automodule:: WindGym.utils.generate_layouts
   :members:
   :undoc-members:

Related Pages
-------------

- `Core Concepts <concepts.md>`_ - Detailed explanations of key concepts
- `Agents <agents.md>`_ - Agent development guide
- `Simulations <simulations.md>`_ - Running simulations
- `Evaluations <evaluations.md>`_ - Evaluation tools and methods
