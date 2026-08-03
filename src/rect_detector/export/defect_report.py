from __future__ import annotations

import json
import importlib
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


def _to_int(value: object, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required mech field: {field_name}")
    return int(value)


def _light_status_from_chip(chip: dict, light_name: str) -> str:
    yolo = chip.get("yolo") or {}
    light_data = yolo.get(light_name)
    if not isinstance(light_data, dict):
        return "OK"
    analyzed = light_data.get("analyzed_output")
    if not isinstance(analyzed, dict):
        return "OK"
    pred_status = str(analyzed.get("pred_status") or "").upper()
    # Requested behavior: missing or INVALID are treated as OK.
    return "NG" if pred_status == "NG" else "OK"


def _collect_light_names(payload: dict) -> list[str]:
    light_names: set[str] = set()
    for sample in payload.get("samples", []):
        for chip in sample.get("chips", []):
            for light_name in (chip.get("yolo") or {}).keys():
                light_names.add(str(light_name))
    return sorted(light_names)


def _build_defect_rows(payload: dict) -> tuple[list[dict], list[str]]:
    light_names = _collect_light_names(payload)
    rows: list[dict] = []

    for sample in payload.get("samples", []):
        for chip in sample.get("chips", []):
            mech = chip.get("mechanical_columns") or {}
            MX = _to_int(mech.get("MX"), "MX")
            MY = _to_int(mech.get("MY"), "MY")

            row = {
                "MX": MX,
                "MY": MY,
            }

            has_ng = False
            for light_name in light_names:
                status = _light_status_from_chip(chip, light_name)
                row[light_name] = status
                if status == "NG":
                    has_ng = True

            row["overall"] = "NG" if has_ng else "OK"
            rows.append(row)

    return rows, light_names


def _merge_xy(
    report_df: pd.DataFrame,
    external_csv_path: Path,
) -> pd.DataFrame:
    def _format_xy_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    external_df = pd.read_csv(external_csv_path)
    column_by_lower_name = {str(column).lower(): column for column in external_df.columns}
    required_columns = ["x", "y", "mx", "my"]
    missing = [column for column in required_columns if column not in column_by_lower_name]
    if missing:
        raise ValueError(
            "External CSV is missing required columns: "
            + ", ".join(column.upper() for column in missing)
        )

    external_df = external_df[
        [column_by_lower_name[column] for column in required_columns]
    ].copy()
    external_df.columns = ["X", "Y", "MX", "MY"]
    external_df["MX"] = external_df["MX"].astype(int)
    external_df["MY"] = external_df["MY"].astype(int)

    duplicate_mask = external_df.duplicated(subset=["MX", "MY"], keep=False)
    if duplicate_mask.any():
        duplicated_rows = external_df.loc[duplicate_mask, ["MX", "MY"]]
        preview = duplicated_rows.drop_duplicates().head(10).to_dict("records")
        raise ValueError(
            "External CSV has duplicated (MX, MY) keys. "
            f"Examples: {preview}"
        )

    merged = report_df.merge(external_df, on=["MX", "MY"], how="left")
    merged = merged[["X", "Y"] + [column for column in merged.columns if column not in ("X", "Y")]]
    for column in ("X", "Y"):
        merged[column] = merged[column].map(_format_xy_value)
    return merged


def export_defect_report(
    *,
    output_json_path: Path | str,
    defect_report_path: Path | str,
    external_xy_csv_path: Path | str | None = None,
    wafer_map_path: Path | str | None = None,
    wafer_map_figsize: tuple[float, float] = (10.0, 8.0),
    wafer_map_chip_aspect: float = 5.0,
) -> Path:
    output_json_path = _resolve_path(output_json_path)
    defect_report_path = _resolve_path(defect_report_path)

    with output_json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rows, light_names = _build_defect_rows(payload)
    report_columns = ["MX", "MY"] + light_names + ["overall"]
    report_df = pd.DataFrame(rows, columns=report_columns)

    if external_xy_csv_path:
        external_xy_csv_path = _resolve_path(external_xy_csv_path)
        report_df = _merge_xy(report_df=report_df, external_csv_path=external_xy_csv_path)
        # Keep only chips that have a valid X/Y mapping in the external CSV.
        # A left merge otherwise emits rows with empty X/Y coordinates.
        report_df = report_df[
            report_df["X"].astype(str).str.strip().ne("")
            & report_df["Y"].astype(str).str.strip().ne("")
        ].copy()

    total_chip_count = len(report_df)
    if total_chip_count == 0:
        yield_rate = 0.0
    else:
        ok_chip_count = int((report_df["overall"] == "OK").sum())
        yield_rate = ok_chip_count / total_chip_count
    report_df["yield_rate"] = round(float(yield_rate), 6)

    if external_xy_csv_path:
        report_df["Bin"] = (report_df["overall"] == "NG").astype(int)

    defect_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(defect_report_path, index=False)

    if external_xy_csv_path:
        if wafer_map_path is None:
            wafer_map_path = defect_report_path.parent.parent / "plots" / "wafer_map_overall.png"
        plot_wafer_map = importlib.import_module(
            "rect_detector.export.wafer_map"
        ).plot_wafer_map
        plot_wafer_map(
            report_df,
            _resolve_path(wafer_map_path),
            figsize=wafer_map_figsize,
            chip_aspect=wafer_map_chip_aspect,
        )

    return defect_report_path
