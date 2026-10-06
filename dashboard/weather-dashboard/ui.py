"""Presentation helpers for the weather dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_header() -> None:
    st.title("Wetterstation")
    st.caption("Aktuelle Messwerte und historische Entwicklungen")


def render_latest_tiles(measurements: list[dict]) -> None:
    if not measurements:
        st.info("Keine aktuellen Messwerte verfügbar.")
        return

    date = pd.to_datetime(measurements[0].get("date"), format="ISO8601", errors="coerce")
    st.markdown(f"{date.strftime('%d.%m.%Y um %H:%M Uhr')}" if date else "Letzte Messwerte")
    max_columns = 3
    num_rows = (len(measurements) + max_columns - 1) // max_columns
    for row in range(num_rows):
        columns = st.columns(max_columns)
        for column, measurement in zip(
            columns, measurements[row * max_columns : (row + 1) * max_columns]
        ):
            value = measurement.get("value")
            formatted_value = f"{value:.1f}" if isinstance(value, (int, float)) else "-"
            with column:
                st.metric(
                    measurement.get("name", "Messwert"),
                    f"{formatted_value} {measurement.get('unit', '')}".strip(),
                )


def render_chart(measurements: list[dict], series_names: list[str]) -> None:
    if not measurements:
        st.info("Für diesen Zeitraum liegen keine Messwerte vor.")
        return

    frame = pd.DataFrame(measurements)
    frame["date"] = pd.to_datetime(frame["date"], format="ISO8601")
    frame["value"] = pd.to_numeric(frame["value"])
    frame["unit"] = frame["unit"].fillna("")
    series_units = frame.groupby("name", sort=False)["unit"].first().to_dict()
    figure = px.line(
        frame,
        x="date",
        y="value",
        color="name",
        custom_data=["unit"],
        labels={"date": "Zeitpunkt", "value": "Messwert", "name": "Messreihe"},
    )

    axis_layout = {}
    axis_by_name = {}
    for index, series_name in enumerate(series_units):
        axis_number = index + 1
        axis_name = "yaxis" if axis_number == 1 else f"yaxis{axis_number}"
        axis_reference = "y" if axis_number == 1 else f"y{axis_number}"
        is_left = index % 2 == 0
        axis_by_name[series_name] = axis_reference
        axis_layout[axis_name] = {
            "title": (
                f"{series_name} ({series_units[series_name]})"
                if series_units[series_name]
                else series_name
            ),
            "overlaying": "y" if axis_number > 1 else None,
            "side": "left" if is_left else "right",
            "position": (0.02 + (index // 2) * 0.06) if is_left else (0.98 - (index // 2) * 0.06),
            "anchor": "free" if axis_number > 1 else "x",
            "showgrid": axis_number == 1,
        }

    for trace in figure.data:
        trace.yaxis = axis_by_name[trace.name]

    figure.update_layout(
        title=" / ".join(series_names),
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        height=420,
        **axis_layout,
    )
    figure.update_traces(
        line={"width": 2.5},
        hovertemplate="%{y:.2f} %{customdata[0]}<extra>%{fullData.name}</extra>",
    )
    st.plotly_chart(figure, width="stretch")


def render_forecast_chart(
    measurements: list[dict],
    forecast: dict[str, list[dict]],
    historical_forecast: dict[str, list[dict]],
    series_names: list[str],
    show_uncertainty: bool,
    show_historical: bool,
) -> None:
    """Render measured data, the current forecast, and an optional past forecast."""
    frame = pd.DataFrame(measurements)
    if frame.empty:
        st.info("Für die Vorhersage liegen keine Messwerte vor.")
        return
    frame["date"] = pd.to_datetime(frame["date"], format="ISO8601")
    frame["value"] = pd.to_numeric(frame["value"])
    series_units = frame.groupby("name", sort=False)["unit"].first().to_dict()
    colors = ("#1769aa", "#d97706", "#16803c")
    figure = go.Figure()
    axis_by_name = {}

    for index, series_name in enumerate(series_names):
        if series_name not in series_units:
            continue
        axis_number = index + 1
        axis_reference = "y" if axis_number == 1 else f"y{axis_number}"
        axis_name = "yaxis" if axis_number == 1 else f"yaxis{axis_number}"
        axis_by_name[series_name] = axis_reference
        color = colors[index % len(colors)]
        measured = frame[frame["name"] == series_name]
        figure.add_trace(
            go.Scatter(
                x=measured["date"],
                y=measured["value"],
                name=series_name,
                mode="lines",
                line={"color": color, "width": 2.5},
                yaxis=axis_reference,
                customdata=[[series_units[series_name]]] * len(measured),
                hovertemplate="%{y:.2f} %{customdata[0]}<extra>%{fullData.name}</extra>",
            )
        )
        current = next(
            (
                entries
                for entries in forecast.values()
                if entries and entries[0]["name"] == series_name
            ),
            [],
        )
        if current:
            current_frame = pd.DataFrame(current)
            current_frame["date"] = pd.to_datetime(current_frame["date"], format="ISO8601")
            current_frame["value"] = pd.to_numeric(current_frame["value"])
            forecast_dates = current_frame["date"].tolist()
            forecast_values = current_frame["value"].tolist()
            last_measured = measured.iloc[-1]
            if last_measured["date"] < forecast_dates[0]:
                forecast_dates.insert(0, last_measured["date"])
                forecast_values.insert(0, last_measured["value"])
            figure.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    name=f"{series_name} (Vorhersage)",
                    mode="lines",
                    line={"color": color, "width": 2.5},
                    yaxis=axis_reference,
                    showlegend=False,
                )
            )
            if show_uncertainty:
                uncertainty = pd.to_numeric(current_frame["uncertainty"])
                figure.add_trace(
                    go.Scatter(
                        x=current_frame["date"],
                        y=current_frame["value"] + 1.96 * uncertainty,
                        mode="lines",
                        line={"width": 0},
                        yaxis=axis_reference,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=current_frame["date"],
                        y=current_frame["value"] - 1.96 * uncertainty,
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(80, 80, 80, 0.14)",
                        yaxis=axis_reference,
                        name="95%-Unsicherheitsintervall",
                        showlegend=index == 0,
                        hoverinfo="skip",
                    )
                )
        if show_historical:
            historic = next(
                (
                    entries
                    for entries in historical_forecast.values()
                    if entries and entries[0]["name"] == series_name
                ),
                [],
            )
            if historic:
                historic_frame = pd.DataFrame(historic)
                historic_frame["date"] = pd.to_datetime(historic_frame["date"], format="ISO8601")
                figure.add_trace(
                    go.Scatter(
                        x=historic_frame["date"],
                        y=historic_frame["value"],
                        name=f"{series_name} (Vergangene Prognose)",
                        mode="lines",
                        line={"color": color, "width": 2.8, "dash": "dash"},
                        opacity=0.9,
                        yaxis=axis_reference,
                        showlegend=True,
                    )
                )

        is_left = index % 2 == 0
        figure.update_layout(
            **{
                axis_name: {
                    "title": f"{series_name} ({series_units[series_name]})",
                    "overlaying": "y" if axis_number > 1 else None,
                    "side": "left" if is_left else "right",
                    "position": (
                        (0.02 + (index // 2) * 0.06) if is_left else (0.98 - (index // 2) * 0.06)
                    ),
                    "anchor": "free" if axis_number > 1 else "x",
                    "showgrid": axis_number == 1,
                }
            }
        )

    figure.update_layout(
        title=" / ".join(series_names),
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        height=420,
        legend={"orientation": "h"},
    )
    if forecast:
        forecast_dates = [
            pd.to_datetime(entry["date"], format="ISO8601")
            for entries in forecast.values()
            for entry in entries
        ]
        figure.add_vline(x=min(forecast_dates), line_width=2, line_color="black")
    st.plotly_chart(figure, width="stretch")
