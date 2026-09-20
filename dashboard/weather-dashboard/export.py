"""Create downloadable exports for weather measurements."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def measurements_frame(measurements: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(measurements, columns=["date", "name", "unit", "value"])
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], format="ISO8601").dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return frame


def create_export(measurements: list[dict], file_format: str) -> tuple[bytes, str, str]:
    frame = measurements_frame(measurements)

    if file_format == "CSV":
        return frame.to_csv(index=False).encode("utf-8-sig"), "text/csv", "csv"
    if file_format == "JSON":
        return frame.to_json(orient="records", date_format="iso", force_ascii=False).encode("utf-8"), "application/json", "json"
    if file_format == "Excel":
        output = BytesIO()
        frame.to_excel(output, index=False, sheet_name="Messwerte")
        return output.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"

    raise ValueError(f"Unbekanntes Dateiformat: {file_format}")
