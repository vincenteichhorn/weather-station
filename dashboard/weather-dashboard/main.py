from datetime import date, timedelta

import streamlit as st

from api import WeatherApi, WeatherApiError
from export import create_export
from ui import render_chart, render_header, render_latest_tiles

st.set_page_config(page_title="Wetterstation", page_icon="☁", layout="wide")


@st.cache_data(ttl=30)
def load_latest() -> list[dict]:
    return WeatherApi().get_latest()


@st.cache_data(ttl=300)
def load_series() -> list[dict]:
    return WeatherApi().get_series()


@st.cache_data(ttl=60)
def load_measurements(series_short: str, start_date: date, end_date: date) -> list[dict]:
    return WeatherApi().get_measurements(series_short, start_date, end_date)


def load_selected_measurements(series_by_name: dict[str, dict], selected_names: list[str], start_date: date, end_date: date) -> list[dict]:
    return [
        measurement for selected_name in selected_names for measurement in load_measurements(series_by_name[selected_name]["shorts"][0], start_date, end_date)
    ]


def main() -> None:
    render_header()

    try:
        latest = load_latest()
        series = [entry for entry in load_series() if not entry.get("disabled", False)]
    except WeatherApiError as error:
        st.error(f"Die Wetter-API ist nicht erreichbar. {error}")
        st.stop()

    st.subheader("Letzte Messwerte")
    render_latest_tiles(latest)

    st.divider()
    st.subheader("Messverlauf")
    if not series:
        st.info("Keine Messreihen verfügbar.")
        return

    controls = st.columns([4, 1, 1])
    series_by_name = {entry["name"]: entry for entry in series}
    selected_names = controls[0].multiselect("Messreihe", list(series_by_name), default=list(series_by_name)[:1])
    start_date = controls[1].date_input("Von", value=date.today() - timedelta(days=7), max_value=date.today())
    end_date = controls[2].date_input("Bis", value=date.today(), min_value=start_date, max_value=date.today())

    if start_date > end_date:
        st.warning("Das Startdatum muss vor dem Enddatum liegen.")
    elif not selected_names:
        st.info("Wähle mindestens eine Messreihe für den Plot aus.")
    else:
        try:
            measurements = load_selected_measurements(series_by_name, selected_names, start_date, end_date)
        except WeatherApiError as error:
            st.error(f"Die Messwerte konnten nicht geladen werden. {error}")
        else:
            render_chart(measurements, selected_names)

    st.divider()
    st.subheader("Messwerte exportieren")
    export_controls = st.columns([4, 1, 1, 1])
    export_names = export_controls[0].multiselect("Messreihen", list(series_by_name), default=selected_names, key="export_series")
    export_start = export_controls[1].date_input("Von", value=start_date, max_value=date.today(), key="export_start")
    export_end = export_controls[2].date_input("Bis", value=end_date, min_value=export_start, max_value=date.today(), key="export_end")
    export_format = export_controls[3].selectbox("Format", ["CSV", "Excel", "JSON"])

    if export_start > export_end:
        st.warning("Das Export-Startdatum muss vor dem Export-Enddatum liegen.")
        return
    if not export_names:
        st.info("Wähle mindestens eine Messreihe für den Export aus.")
        return

    try:
        export_measurements = load_selected_measurements(series_by_name, export_names, export_start, export_end)
        export_bytes, mime_type, extension = create_export(export_measurements, export_format)
    except (WeatherApiError, ValueError) as error:
        st.error(f"Der Export konnte nicht erstellt werden. {error}")
        return

    filename = f"wetterstation_{export_start.isoformat()}_{export_end.isoformat()}.{extension}"
    st.download_button(
        "Datei herunterladen",
        data=export_bytes,
        file_name=filename,
        mime=mime_type,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
