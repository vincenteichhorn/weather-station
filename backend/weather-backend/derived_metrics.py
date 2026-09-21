"""Definitions and calculations for virtual weather measurements."""

from collections.abc import Callable
from dataclasses import dataclass
import math

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
    if not 0 <= humidity <= 100:
        raise ValueError("humidity must be between 0 and 100 percent")

    t = temperature * 9 / 5 + 32  # the NWS regression works in Fahrenheit
    rh = humidity

    # The heat index is only defined for warm air (>= 80 °F, about 26.7 °C).
    if t < 80:
        return temperature

    # Rothfusz regression used by the U.S. National Weather Service.
    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * rh
        - 0.22475541 * t * rh
        - 6.83783e-3 * t**2
        - 5.481717e-2 * rh**2
        + 1.22874e-3 * t**2 * rh
        + 8.5282e-4 * t * rh**2
        - 1.99e-6 * t**2 * rh**2
    )

    # NWS corrections for the edges of the regression's valid range.
    if rh < 13 and t <= 112:
        hi -= ((13 - rh) / 4) * math.sqrt((17 - abs(t - 95)) / 17)
    elif rh > 85 and t <= 87:
        hi += ((rh - 85) / 10) * ((87 - t) / 5)

    # In hot conditions it should never feel cooler than the air temperature.
    hi = max(hi, t)

    return round((hi - 32) * 5 / 9, 1)


DERIVED_METRICS = (
    DerivedMetric("Taupunkt", "°C", ("dewpoint", "dew"), calculate_dew_point),
    DerivedMetric("Absolute Feuchtigkeit", "g/m³", ("abshum", "absolutehumidity"), calculate_absolute_humidity),
    DerivedMetric("Gefühlte Temperatur", "°C", ("realtemp", "heatindex"), calculate_real_temperature),
)

DERIVED_METRICS_BY_SHORT = {short: metric for metric in DERIVED_METRICS for short in metric.shorts}
