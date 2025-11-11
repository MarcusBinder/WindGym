"""
Configuration schema for WindGym environments.

This module provides strongly-typed configuration classes with validation
for all WindGym environment parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
import warnings


@dataclass
class FarmConfig:
    """Configuration for wind farm layout and turbine constraints."""

    yaw_min: float
    yaw_max: float
    xDist: Optional[float] = None  # Optional: for grid layouts
    yDist: Optional[float] = None  # Optional: for grid layouts
    nx: Optional[int] = None  # Optional: for grid layouts
    ny: Optional[int] = None  # Optional: for grid layouts

    def __post_init__(self):
        """Validate farm configuration."""
        if self.yaw_min >= self.yaw_max:
            raise ValueError(
                f"yaw_min ({self.yaw_min}) must be less than yaw_max ({self.yaw_max})"
            )
        if self.yaw_min < -180 or self.yaw_max > 180:
            raise ValueError(
                f"Yaw angles must be in range [-180, 180], got [{self.yaw_min}, {self.yaw_max}]"
            )


@dataclass
class WindConfig:
    """Configuration for wind conditions sampling."""

    ws_min: float  # Minimum wind speed (m/s)
    ws_max: float  # Maximum wind speed (m/s)
    TI_min: float  # Minimum turbulence intensity (0-1)
    TI_max: float  # Maximum turbulence intensity (0-1)
    wd_min: float  # Minimum wind direction (degrees)
    wd_max: float  # Maximum wind direction (degrees)

    def __post_init__(self):
        """Validate wind configuration."""
        if self.ws_min > self.ws_max:
            raise ValueError(
                f"ws_min ({self.ws_min}) must be less than or equal to ws_max ({self.ws_max})"
            )
        if self.ws_min < 0:
            raise ValueError(f"ws_min must be non-negative, got {self.ws_min}")
        if self.ws_max > 50:
            warnings.warn(
                f"ws_max of {self.ws_max} m/s is unusually high. Typical values are 3-25 m/s."
            )

        if self.TI_min > self.TI_max:
            raise ValueError(
                f"TI_min ({self.TI_min}) must be less than or equal to TI_max ({self.TI_max})"
            )
        if not (0 <= self.TI_min <= 1):
            raise ValueError(f"TI_min must be in [0, 1], got {self.TI_min}")
        if not (0 <= self.TI_max <= 1):
            raise ValueError(f"TI_max must be in [0, 1], got {self.TI_max}")

        if self.wd_min > self.wd_max:
            raise ValueError(
                f"wd_min ({self.wd_min}) must be less than or equal to wd_max ({self.wd_max})"
            )
        if self.wd_min < 0 or self.wd_max > 360:
            raise ValueError(
                f"Wind direction must be in [0, 360], got [{self.wd_min}, {self.wd_max}]"
            )


@dataclass
class ActionPenaltyConfig:
    """Configuration for action penalty in reward calculation."""

    action_penalty: float = 0.0
    action_penalty_type: Literal["Change", "Absolute", "None"] = "Change"

    def __post_init__(self):
        """Validate action penalty configuration."""
        if self.action_penalty < 0:
            raise ValueError(
                f"action_penalty must be non-negative, got {self.action_penalty}"
            )
        valid_types = {"Change", "Absolute", "None"}
        if self.action_penalty_type not in valid_types:
            raise ValueError(
                f"action_penalty_type must be one of {valid_types}, got {self.action_penalty_type}"
            )


@dataclass
class PowerRewardConfig:
    """Configuration for power reward calculation."""

    Power_reward: Literal["Baseline", "Power_avg", "Power_diff", "None"] = "Baseline"
    Power_avg: int = 10  # Window size for averaging
    Power_scaling: float = 1.0

    def __post_init__(self):
        """Validate power reward configuration."""
        valid_rewards = {"Baseline", "Power_avg", "Power_diff", "None"}
        if self.Power_reward not in valid_rewards:
            raise ValueError(
                f"Power_reward must be one of {valid_rewards}, got {self.Power_reward}"
            )
        if self.Power_avg < 1:
            raise ValueError(
                f"Power_avg must be at least 1, got {self.Power_avg}"
            )
        if self.Power_avg > 1000:
            warnings.warn(
                f"Power_avg of {self.Power_avg} is very large and may cause memory issues"
            )
        if self.Power_scaling <= 0:
            raise ValueError(
                f"Power_scaling must be positive, got {self.Power_scaling}"
            )


@dataclass
class MeasurementLevelConfig:
    """Configuration for which measurements are observed."""

    turb_ws: bool = False
    turb_wd: bool = False
    turb_TI: bool = False
    turb_power: bool = False
    farm_ws: bool = False
    farm_wd: bool = False
    farm_TI: bool = False
    farm_power: bool = False
    ti_sample_count: int = 30  # Number of samples for TI estimation

    def __post_init__(self):
        """Validate measurement level configuration."""
        if self.ti_sample_count < 1:
            raise ValueError(
                f"ti_sample_count must be at least 1, got {self.ti_sample_count}"
            )


@dataclass
class MeasurementDetailsConfig:
    """Configuration for measurement history and processing."""

    current: bool = False  # Include current value
    rolling_mean: bool = False  # Include rolling mean
    history_N: int = 1  # Number of history samples
    history_length: int = 10  # Length of history window
    window_length: int = 10  # Length of rolling mean window

    def __post_init__(self):
        """Validate measurement details configuration."""
        if self.history_N < 0:
            raise ValueError(f"history_N must be non-negative, got {self.history_N}")
        if self.history_length < 1:
            raise ValueError(
                f"history_length must be at least 1, got {self.history_length}"
            )
        if self.window_length < 1:
            raise ValueError(
                f"window_length must be at least 1, got {self.window_length}"
            )


@dataclass
class ProbeConfig:
    """Configuration for a single wind speed probe."""

    name: str
    turbine_index: int
    relative_position: List[float]  # [x, y, z] relative to turbine
    include_wakes: bool = True
    probe_type: Literal["WS", "WD", "TI"] = "WS"

    def __post_init__(self):
        """Validate probe configuration."""
        if len(self.relative_position) != 3:
            raise ValueError(
                f"relative_position must have 3 elements [x, y, z], got {self.relative_position}"
            )
        if self.turbine_index < 0:
            raise ValueError(
                f"turbine_index must be non-negative, got {self.turbine_index}"
            )


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""

    dt_sim: float = 1.0  # Simulation timestep in seconds
    dt_env: float = 1.0  # Environment timestep in seconds
    n_passthrough: int = 5  # Number of flow passthroughs
    burn_in_passthroughs: int = 2  # Burn-in passthroughs before episode starts
    max_turb_move: float = 2.0  # Maximum turbine movement per timestep (for DWM stability)

    def __post_init__(self):
        """Validate simulation configuration."""
        if self.dt_sim <= 0:
            raise ValueError(f"dt_sim must be positive, got {self.dt_sim}")
        if self.dt_env <= 0:
            raise ValueError(f"dt_env must be positive, got {self.dt_env}")
        if self.dt_env % self.dt_sim != 0:
            raise ValueError(
                f"dt_env ({self.dt_env}) must be a multiple of dt_sim ({self.dt_sim})"
            )
        if self.n_passthrough < 1:
            raise ValueError(
                f"n_passthrough must be at least 1, got {self.n_passthrough}"
            )
        if self.burn_in_passthroughs < 0:
            raise ValueError(
                f"burn_in_passthroughs must be non-negative, got {self.burn_in_passthroughs}"
            )
        if self.max_turb_move <= 0:
            raise ValueError(
                f"max_turb_move must be positive, got {self.max_turb_move}"
            )


@dataclass
class ScalingConfig:
    """Configuration for observation and action space scaling."""

    ws_min: float = 0.0
    ws_max: float = 30.0
    wd_min: float = 0.0
    wd_max: float = 360.0
    ti_min: float = 0.0
    ti_max: float = 1.0
    yaw_min: float = -45.0
    yaw_max: float = 45.0

    def __post_init__(self):
        """Validate scaling configuration."""
        if self.ws_min >= self.ws_max:
            raise ValueError(
                f"ws_min ({self.ws_min}) must be less than ws_max ({self.ws_max})"
            )
        if self.wd_min >= self.wd_max:
            raise ValueError(
                f"wd_min ({self.wd_min}) must be less than wd_max ({self.wd_max})"
            )
        if self.ti_min >= self.ti_max:
            raise ValueError(
                f"ti_min ({self.ti_min}) must be less than ti_max ({self.ti_max})"
            )
        if self.yaw_min >= self.yaw_max:
            raise ValueError(
                f"yaw_min ({self.yaw_min}) must be less than yaw_max ({self.yaw_max})"
            )


@dataclass
class EnvConfig:
    """Complete environment configuration with all subsections."""

    # Required sections (must come first in dataclass)
    farm: FarmConfig
    wind: WindConfig
    mes_level: MeasurementLevelConfig
    ws_mes: MeasurementDetailsConfig
    wd_mes: MeasurementDetailsConfig
    yaw_mes: MeasurementDetailsConfig
    power_mes: MeasurementDetailsConfig

    # Optional sections with defaults
    act_pen: ActionPenaltyConfig = field(default_factory=ActionPenaltyConfig)
    power_def: PowerRewardConfig = field(default_factory=PowerRewardConfig)

    # Top-level optional fields
    version: str = "1.0"
    yaw_init: Literal["Random", "Defined", "Zeros"] = "Random"
    noise: str = "None"
    BaseController: Literal["Local", "Global", "PyWake"] = "Local"
    ActionMethod: Literal["yaw", "wind", "absolute"] = "wind"
    Track_power: bool = False

    # Probes (optional)
    probes: List[ProbeConfig] = field(default_factory=list)

    def __post_init__(self):
        """Validate environment configuration."""
        valid_yaw_inits = {"Random", "Defined", "Zeros"}
        if self.yaw_init not in valid_yaw_inits:
            raise ValueError(
                f"yaw_init must be one of {valid_yaw_inits}, got {self.yaw_init}"
            )

        valid_base_controllers = {"Local", "Global", "PyWake"}
        if self.BaseController not in valid_base_controllers:
            raise ValueError(
                f"BaseController must be one of {valid_base_controllers}, got {self.BaseController}"
            )

        valid_action_methods = {"yaw", "wind", "absolute"}
        if self.ActionMethod not in valid_action_methods:
            raise ValueError(
                f"ActionMethod must be one of {valid_action_methods}, got {self.ActionMethod}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> EnvConfig:
        """
        Create EnvConfig from a dictionary (typically loaded from YAML).

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            Validated EnvConfig instance

        Raises:
            ValueError: If required sections or keys are missing or invalid
        """
        # Helper to ensure section exists
        def get_section(name: str, required: bool = True) -> Dict[str, Any]:
            section = config_dict.get(name, {})
            if required and not isinstance(section, dict):
                raise ValueError(
                    f"Config section '{name}' is required and must be a mapping."
                )
            return section or {}

        # Parse farm config
        farm_dict = get_section("farm")
        farm = FarmConfig(**farm_dict)

        # Parse wind config
        wind_dict = get_section("wind")
        wind = WindConfig(**wind_dict)

        # Parse measurement level
        mes_level_dict = get_section("mes_level")
        mes_level = MeasurementLevelConfig(**mes_level_dict)

        # Parse measurement details
        ws_mes_dict = get_section("ws_mes")
        ws_mes = MeasurementDetailsConfig(
            current=ws_mes_dict.get("ws_current", False),
            rolling_mean=ws_mes_dict.get("ws_rolling_mean", False),
            history_N=ws_mes_dict.get("ws_history_N", 1),
            history_length=ws_mes_dict.get("ws_history_length", 10),
            window_length=ws_mes_dict.get("ws_window_length", 10),
        )

        wd_mes_dict = get_section("wd_mes")
        wd_mes = MeasurementDetailsConfig(
            current=wd_mes_dict.get("wd_current", False),
            rolling_mean=wd_mes_dict.get("wd_rolling_mean", False),
            history_N=wd_mes_dict.get("wd_history_N", 1),
            history_length=wd_mes_dict.get("wd_history_length", 10),
            window_length=wd_mes_dict.get("wd_window_length", 10),
        )

        yaw_mes_dict = get_section("yaw_mes")
        yaw_mes = MeasurementDetailsConfig(
            current=yaw_mes_dict.get("yaw_current", False),
            rolling_mean=yaw_mes_dict.get("yaw_rolling_mean", False),
            history_N=yaw_mes_dict.get("yaw_history_N", 1),
            history_length=yaw_mes_dict.get("yaw_history_length", 10),
            window_length=yaw_mes_dict.get("yaw_window_length", 10),
        )

        power_mes_dict = get_section("power_mes")
        power_mes = MeasurementDetailsConfig(
            current=power_mes_dict.get("power_current", False),
            rolling_mean=power_mes_dict.get("power_rolling_mean", False),
            history_N=power_mes_dict.get("power_history_N", 1),
            history_length=power_mes_dict.get("power_history_length", 10),
            window_length=power_mes_dict.get("power_window_length", 10),
        )

        # Parse optional sections
        act_pen_dict = get_section("act_pen", required=False)
        act_pen = ActionPenaltyConfig(**act_pen_dict) if act_pen_dict else ActionPenaltyConfig()

        power_def_dict = get_section("power_def", required=False)
        power_def = PowerRewardConfig(**power_def_dict) if power_def_dict else PowerRewardConfig()

        # Parse probes
        probes_list = config_dict.get("probes", [])
        probes = [ProbeConfig(**probe_dict) for probe_dict in probes_list]

        # Top-level fields
        return cls(
            version=config_dict.get("version", "1.0"),
            farm=farm,
            wind=wind,
            mes_level=mes_level,
            ws_mes=ws_mes,
            wd_mes=wd_mes,
            yaw_mes=yaw_mes,
            power_mes=power_mes,
            act_pen=act_pen,
            power_def=power_def,
            yaw_init=config_dict.get("yaw_init", "Random"),
            noise=config_dict.get("noise", "None"),
            BaseController=config_dict.get("BaseController", "Local"),
            ActionMethod=config_dict.get("ActionMethod", "wind"),
            Track_power=config_dict.get("Track_power", False),
            probes=probes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert EnvConfig to a dictionary suitable for YAML export.

        Returns:
            Dictionary representation of the configuration
        """
        from dataclasses import asdict

        return asdict(self)
