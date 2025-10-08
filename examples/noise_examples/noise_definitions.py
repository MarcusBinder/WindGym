from WindGym.Measurement_Manager import (
    HybridNoiseModel,
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    MeasurementType,
)


def create_procedural_noise_model(
    ws_std: float = 0.5,
    wd_std: float = 2.0,
    ws_bias_range: tuple[float, float] = (-2.0, 2.0),
    wd_bias_range: tuple[float, float] = (-10.0, 10.0),
) -> HybridNoiseModel:
    """
    Factory function for the standardized "Procedural Noise" model.
    """
    white_noise = WhiteNoiseModel(
        noise_std_devs={
            MeasurementType.WIND_SPEED: ws_std,
            MeasurementType.WIND_DIRECTION: wd_std,
        }
    )

    bias_noise = EpisodicBiasNoiseModel(
        {
            MeasurementType.WIND_SPEED: (-4.0, 4.0),  # Bias of up to +/- 4 m/s
            MeasurementType.WIND_DIRECTION: (
                -10.0,
                10.0,
            ),  # Bias of up to +/- 45 degrees
            MeasurementType.POWER: (
                -500000.0,
                500000.0,
            ),  # Bias of up to +/- 500 kW
            MeasurementType.YAW_ANGLE: (
                -20.0,
                20.0,
            ),  # Bias of up to +/- 10 degrees
        }
    )

    return HybridNoiseModel(models=[white_noise, bias_noise])
