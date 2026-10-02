"""Reusable weather chart construction helpers."""

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def add_daylight_background(axis, start_date, end_date, sun_times):
    """Add twilight and nighttime spans without filling any data series."""
    if not sun_times:
        return

    axis.axvspan(
        start_date,
        sun_times["civil_twilight_begin"],
        facecolor="#657384",
        alpha=0.24,
        zorder=0,
    )
    axis.axvspan(
        sun_times["civil_twilight_begin"],
        sun_times["sunrise"],
        facecolor="#aeb8c4",
        alpha=0.18,
        zorder=0,
    )
    axis.axvspan(
        sun_times["sunset"],
        sun_times["civil_twilight_end"],
        facecolor="#aeb8c4",
        alpha=0.18,
        zorder=0,
    )
    axis.axvspan(
        sun_times["civil_twilight_end"],
        end_date,
        facecolor="#657384",
        alpha=0.24,
        zorder=0,
    )


def create_weather_plot(plots, start_date, end_date, title, sun_times=None):
    """Create a multi-axis weather plot from named measurement series."""
    figure, base_axis = plt.subplots(figsize=(12, 5.5), facecolor="#f4f7fb")
    base_axis.set_facecolor("#ffffff")
    add_daylight_background(base_axis, start_date, end_date, sun_times)

    axes = [base_axis]
    for _ in range(1, len(plots)):
        axes.append(base_axis.twinx())

    colors = ("#1769aa", "#d97706", "#16803c")
    lines = []
    labels = []
    for index, (axis, (name, unit, measurements)) in enumerate(zip(axes, plots)):
        dates = [datetime.fromisoformat(measurement["date"]) for measurement in measurements]
        values = [measurement["value"] for measurement in measurements]
        color = colors[index % len(colors)]
        axis.set_facecolor("none")
        (line,) = axis.plot(
            dates,
            values,
            color=color,
            linewidth=3,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        axis.margins(y=0.18)
        minimum_index = values.index(min(values))
        maximum_index = values.index(max(values))
        axis.scatter(
            [dates[minimum_index]],
            [values[minimum_index]],
            color="#ffffff",
            edgecolor=color,
            linewidth=2,
            s=80,
            zorder=5,
            clip_on=True,
        )
        axis.scatter(
            [dates[maximum_index]],
            [values[maximum_index]],
            color=color,
            edgecolor="#ffffff",
            linewidth=1.5,
            marker="D",
            s=70,
            zorder=5,
            clip_on=True,
        )
        for value_index, y_offset, vertical_alignment in (
            (minimum_index, -10, "top"),
            (maximum_index, 10, "bottom"),
        ):
            axis.annotate(
                f"{values[value_index]:.2f} {unit}",
                (dates[value_index], values[value_index]),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=10,
                fontweight="bold",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
                annotation_clip=True,
            )
        lines.append(line)
        labels.append(name)
        axis.set_ylabel(f"{name} ({unit})", color=color, fontsize=14)
        axis.tick_params(axis="y", colors=color, labelsize=12, width=1.5)
        spine = "left" if index == 0 else "right"
        axis.spines[spine].set_color(color)
        axis.spines[spine].set_linewidth(2)
        axis.spines["top"].set_visible(False)
        if index == 0:
            axis.spines["right"].set_visible(False)
        else:
            axis.spines["left"].set_visible(False)
        if index > 1:
            axis.spines["right"].set_position(("axes", 1.12 + (index - 2) * 0.1))

    base_axis.set_title(title, fontsize=20, pad=22, fontweight="bold", loc="left", color="#172033")
    base_axis.set_xlabel("Zeit", fontsize=14)
    base_axis.tick_params(axis="x", labelsize=12, width=1.5, colors="#526176")
    base_axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    base_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))
    base_axis.grid(True, axis="y", color="#dbe3ed", alpha=0.8, linewidth=1)
    base_axis.grid(True, axis="x", color="#edf1f5", alpha=0.9, linewidth=0.8)
    base_axis.spines["bottom"].set_linewidth(1.5)
    base_axis.spines["bottom"].set_color("#9aa8b8")
    base_axis.set_xlim(start_date - timedelta(minutes=15), end_date + timedelta(minutes=15))
    legend_handles = lines
    legend_labels = labels
    if sun_times:
        legend_handles = [
            *lines,
            Patch(facecolor="#657384", edgecolor="none", alpha=0.24),
        ]
        legend_labels = [*labels, "Dämmerung/Nacht"]
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure


def create_forecast_plot(plots, start_date, forecast_start, end_date, title, sun_times=None):
    """Create a measured-and-forecast plot with per-series uncertainty bands."""
    figure, base_axis = plt.subplots(figsize=(12, 5.5), facecolor="#f4f7fb")
    base_axis.set_facecolor("#ffffff")
    add_daylight_background(base_axis, start_date, end_date, sun_times)

    axes = [base_axis]
    for _ in range(1, len(plots)):
        axes.append(base_axis.twinx())

    colors = ("#1769aa", "#d97706", "#16803c")
    lines = []
    labels = []
    for index, (axis, (name, unit, measured, forecast)) in enumerate(zip(axes, plots)):
        color = colors[index % len(colors)]
        measured_dates = [datetime.fromisoformat(entry["date"]) for entry in measured]
        measured_values = [entry["value"] for entry in measured]
        forecast_dates = [datetime.fromisoformat(entry["date"]) for entry in forecast]
        forecast_values = [entry["value"] for entry in forecast]
        uncertainties = [entry["uncertainty"] for entry in forecast]

        axis.set_facecolor("none")
        (line,) = axis.plot(
            measured_dates,
            measured_values,
            color=color,
            linewidth=3,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        axis.plot(
            [measured_dates[-1], *forecast_dates],
            [measured_values[-1], *forecast_values],
            color=color,
            linewidth=3,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        axis.fill_between(
            forecast_dates,
            [
                value - 1.96 * uncertainty
                for value, uncertainty in zip(forecast_values, uncertainties)
            ],
            [
                value + 1.96 * uncertainty
                for value, uncertainty in zip(forecast_values, uncertainties)
            ],
            color=color,
            alpha=0.14,
            zorder=2,
        )
        axis.margins(y=0.18)
        minimum_index = measured_values.index(min(measured_values))
        maximum_index = measured_values.index(max(measured_values))
        axis.scatter(
            [measured_dates[minimum_index]],
            [measured_values[minimum_index]],
            color="#ffffff",
            edgecolor=color,
            linewidth=2,
            s=80,
            zorder=5,
        )
        axis.scatter(
            [measured_dates[maximum_index]],
            [measured_values[maximum_index]],
            color=color,
            edgecolor="#ffffff",
            linewidth=1.5,
            marker="D",
            s=70,
            zorder=5,
        )
        for value_index, y_offset, vertical_alignment in (
            (minimum_index, -10, "top"),
            (maximum_index, 10, "bottom"),
        ):
            axis.annotate(
                f"{measured_values[value_index]:.2f} {unit}",
                (measured_dates[value_index], measured_values[value_index]),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=10,
                fontweight="bold",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

        forecast_minimum_index = forecast_values.index(min(forecast_values))
        forecast_maximum_index = forecast_values.index(max(forecast_values))
        axis.scatter(
            [forecast_dates[forecast_minimum_index]],
            [forecast_values[forecast_minimum_index]],
            color="#ffffff",
            edgecolor=color,
            linewidth=2,
            marker="v",
            s=80,
            zorder=5,
        )
        axis.scatter(
            [forecast_dates[forecast_maximum_index]],
            [forecast_values[forecast_maximum_index]],
            color=color,
            edgecolor="#ffffff",
            linewidth=1.5,
            marker="^",
            s=70,
            zorder=5,
        )
        for value_index, y_offset, vertical_alignment in (
            (forecast_minimum_index, -10, "top"),
            (forecast_maximum_index, 10, "bottom"),
        ):
            axis.annotate(
                f"{forecast_values[value_index]:.2f} {unit}",
                (forecast_dates[value_index], forecast_values[value_index]),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=10,
                fontweight="bold",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

        lines.append(line)
        labels.append(name)
        axis.set_ylabel(f"{name} ({unit})", color=color, fontsize=14)
        axis.tick_params(axis="y", colors=color, labelsize=12, width=1.5)
        spine = "left" if index == 0 else "right"
        axis.spines[spine].set_color(color)
        axis.spines[spine].set_linewidth(2)
        axis.spines["top"].set_visible(False)
        if index == 0:
            axis.spines["right"].set_visible(False)
        else:
            axis.spines["left"].set_visible(False)
        if index > 1:
            axis.spines["right"].set_position(("axes", 1.12 + (index - 2) * 0.1))

    base_axis.axvline(forecast_start, color="black", linewidth=2.5, zorder=6)
    base_axis.set_title(title, fontsize=20, pad=22, fontweight="bold", loc="left", color="#172033")
    base_axis.set_xlabel("Zeit", fontsize=14)
    base_axis.tick_params(axis="x", labelsize=12, width=1.5, colors="#526176")
    base_axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    base_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))
    base_axis.grid(True, axis="y", color="#dbe3ed", alpha=0.8, linewidth=1)
    base_axis.grid(True, axis="x", color="#edf1f5", alpha=0.9, linewidth=0.8)
    base_axis.spines["bottom"].set_linewidth(1.5)
    base_axis.spines["bottom"].set_color("#9aa8b8")
    base_axis.set_xlim(start_date - timedelta(minutes=15), end_date + timedelta(minutes=15))
    handles = [*lines, Patch(facecolor="#7f8c8d", edgecolor="#7f8c8d", alpha=0.14)]
    legend_labels = [*labels, "95%-Unsicherheitsintervall"]
    if sun_times:
        handles.append(Patch(facecolor="#657384", edgecolor="none", alpha=0.24))
        legend_labels.append("Dämmerung/Nacht")
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(handles),
        frameon=False,
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure
