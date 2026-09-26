"""Presentation helpers for the weather dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
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
        for column, measurement in zip(columns, measurements[row * max_columns : (row + 1) * max_columns]):
            value = measurement.get("value")
            formatted_value = f"{value:.1f}" if isinstance(value, (int, float)) else "-"
            with column:
                st.metric(measurement.get("name", "Messwert"), f"{formatted_value} {measurement.get('unit', '')}".strip())


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
        markers=True,
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
            "title": f"{series_name} ({series_units[series_name]})" if series_units[series_name] else series_name,
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
    figure.update_traces(line={"width": 2.5}, hovertemplate="%{y:.2f} %{customdata[0]}<extra>%{fullData.name}</extra>")
    st.plotly_chart(figure, width="stretch")
