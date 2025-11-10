"""
Core modules for WindGym environment.

This package contains modular components that handle specific responsibilities
of the wind farm environment, promoting separation of concerns and maintainability.
"""

from .reward_calculator import RewardCalculator

__all__ = ["RewardCalculator"]
