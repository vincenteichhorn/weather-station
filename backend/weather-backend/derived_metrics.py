"""Definitions and calculations for virtual weather measurements."""

from collections.abc import Callable
from dataclasses import dataclass

MetricFunction = Callable[[float, float], float]
"""Callable type used to calculate a derived metric from temperature and humidity."""


@dataclass(frozen=True)
class DerivedMetric:
    """Describe a derived metric exposed by the API."""

    name: str
    unit: str
    shorts: tuple[str, ...]
    calculate: MetricFunction


def calculate_dew_point(temperature: float, humidity: float) -> float:
    """Calculate the dew point in degrees Celsius.

    Args:
        temperature: Air temperature in degrees Celsius.
        humidity: Relative humidity in percent.

    Returns:
        The estimated dew point in degrees Celsius.
    """
    return temperature - ((100 - humidity) / 5)


def calculate_absolute_humidity(temperature: float, humidity: float) -> float:
    """Calculate absolute humidity in grams per cubic metre.

    Args:
        temperature: Air temperature in degrees Celsius.
        humidity: Relative humidity in percent.

    Returns:
        The estimated absolute humidity in grams per cubic metre.
    """
    temperature_kelvin = temperature + 273.15
    vapor_pressure = 6.112 * (10 ** ((7.5 * temperature) / (237.7 + temperature))) * (humidity / 100)
    return (vapor_pressure * 100) / (461.5 * temperature_kelvin) * 1000


def calculate_real_temperature(temperature: float, humidity: float) -> float:
    """Calculate an estimated perceived temperature.

    The heat-index formula is only meaningful for warm conditions. A weather
    station without wind speed cannot calculate a reliable wind-chill value,
    so cold temperatures are reported unchanged instead of producing a
    misleading result from the heat-index polynomial.

    Args:
        temperature: Air temperature in degrees Celsius.
        humidity: Relative humidity in percent.

    Returns:
        The estimated perceived temperature in degrees Celsius.
    """
    if temperature < 20:
        return temperature

    return (
        -8.784695
        + 1.61139411 * temperature
        + 2.338549 * humidity
        - 0.14611605 * temperature * humidity
        - 0.012308094 * (temperature**2)
        - 0.016424828 * (humidity**2)
        + 0.002211732 * (temperature**2) * humidity
        + 0.00072546 * temperature * (humidity**2)
        - 0.000003582 * (temperature**2) * (humidity**2)
    )


DERIVED_METRICS = (
    DerivedMetric("Taupunkt", "°C", ("dewpoint", "dew"), calculate_dew_point),
    DerivedMetric("Absolute Feuchtigkeit", "g/m³", ("abshum", "absolutehumidity"), calculate_absolute_humidity),
    DerivedMetric("Gefühlte Temperatur", "°C", ("realtemp", "heatindex"), calculate_real_temperature),
)

DERIVED_METRICS_BY_SHORT = {short: metric for metric in DERIVED_METRICS for short in metric.shorts}
