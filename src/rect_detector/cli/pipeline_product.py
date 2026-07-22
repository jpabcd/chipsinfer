from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from rect_detector.cli.infer_dataloader import main as infer_dataloader_main


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run rect_detector pipeline on one product directory."
    )
    parser.add_argument(
        "product_dir",
        nargs="?",
        default="../../48AMA/imgs/S26F20082-02",
        help="Product directory relative to Detectors/rect_detector, e.g. ../../48AMA/imgs/S26F20082-02",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--light-read-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base directory for all generated outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON path, relative to current working directory.",
    )
    parser.add_argument(
        "--yolo-config",
        type=str,
        default="combined_yolo_inferers_config.example.json",
        help="YOLO config path. Relative paths are resolved from current working directory.",
    )
    parser.add_argument("--x-scale", type=float, default=0.30)
    parser.add_argument("--y-scale", type=float, default=0.50)
    parser.add_argument("--mech-delta-x", type=float, default=1421.0)
    parser.add_argument("--mech-delta-y", type=float, default=283.0)
    parser.add_argument("--delta-x-pixel", type=float, default=950.0)
    parser.add_argument("--delta-y-pixel", type=float, default=185.0)
    parser.add_argument("--strict-align", action="store_true")
    parser.add_argument("--unsafe-missing-ok", action="store_true")
    parser.add_argument("--trace-batches", action="store_true")
    parser.add_argument(
        "--save-predict-input",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--predict-input-root", type=str, default="")
    parser.add_argument(
        "--defect-report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export per-chip defect report CSV with per-light OK/NG and overall OK/NG.",
    )
    parser.add_argument(
        "--defect-report-path",
        type=str,
        default="",
        help="Defect report CSV output path.",
    )
    parser.add_argument(
        "--external-xy-csv-path",
        type=str,
        default="",
        help="Optional CSV path with columns X,Y,Mx,My for left merge by (Mx,My).",
    )
    parser.add_argument(
        "--wafer-map-path",
        type=str,
        default="",
        help="Wafer map PNG output path. Defaults to output-dir/plots/wafer_map_overall.png.",
    )
    parser.add_argument(
        "--wafer-map-figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(10.0, 8.0),
        help="Wafer map figure size in inches: WIDTH HEIGHT. Default: 10 8.",
    )
    parser.add_argument(
        "--wafer-map-chip-aspect",
        type=float,
        default=5.0,
        help="Chip physical width/height ratio used for wafer-map geometry. Default: 5.0.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_json_path = Path(args.output_json) if args.output_json else output_dir / "json" / "main_inferer_dataloader_results.json"
    predict_input_root = Path(args.predict_input_root) if args.predict_input_root else output_dir / "predict_input"
    defect_report_path = Path(args.defect_report_path) if args.defect_report_path else output_dir / "csv" / "defect_report.csv"
    wafer_map_path = Path(args.wafer_map_path) if args.wafer_map_path else output_dir / "plots" / "wafer_map_overall.png"

    infer_args = [
        str(Path(args.product_dir)),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--light-read-workers",
        str(args.light_read_workers),
        "--max-batches",
        str(args.max_batches),
        "--max-samples",
        str(args.max_samples),
        "--output-json",
        str(output_json_path),
        "--yolo-config",
        str(Path(args.yolo_config)),
        "--x-scale",
        str(args.x_scale),
        "--y-scale",
        str(args.y_scale),
        "--mech-delta-x",
        str(args.mech_delta_x),
        "--mech-delta-y",
        str(args.mech_delta_y),
        "--delta-x-pixel",
        str(args.delta_x_pixel),
        "--delta-y-pixel",
        str(args.delta_y_pixel),
        "--predict-input-root",
        str(predict_input_root),
    ]

    if args.persistent_workers:
        infer_args.append("--persistent-workers")
    else:
        infer_args.append("--no-persistent-workers")

    if args.strict_align:
        infer_args.append("--strict-align")
    if args.unsafe_missing_ok:
        infer_args.append("--unsafe-missing-ok")
    if args.trace_batches:
        infer_args.append("--trace-batches")
    if args.save_predict_input:
        infer_args.append("--save-predict-input")
    else:
        infer_args.append("--no-save-predict-input")

    infer_dataloader_main(infer_args)

    if args.defect_report or args.external_xy_csv_path.strip():
        export_defect_report = importlib.import_module(
            "rect_detector.export.defect_report"
        ).export_defect_report

        project_root = _project_root()
        output_json_path = Path(output_json_path)
        if not output_json_path.is_absolute():
            output_json_path = (project_root / output_json_path).resolve()

        defect_report_path = Path(defect_report_path)
        if not defect_report_path.is_absolute():
            defect_report_path = (project_root / defect_report_path).resolve()

        wafer_map_path = Path(wafer_map_path)
        if not wafer_map_path.is_absolute():
            wafer_map_path = (project_root / wafer_map_path).resolve()

        external_xy_csv_path = args.external_xy_csv_path.strip() or None
        if external_xy_csv_path is not None:
            external_path_obj = Path(external_xy_csv_path)
            if not external_path_obj.is_absolute():
                external_xy_csv_path = str((project_root / external_path_obj).resolve())

        generated_path = export_defect_report(
            output_json_path=output_json_path,
            defect_report_path=defect_report_path,
            external_xy_csv_path=external_xy_csv_path,
            wafer_map_path=wafer_map_path if external_xy_csv_path else None,
            wafer_map_figsize=tuple(args.wafer_map_figsize),
            wafer_map_chip_aspect=args.wafer_map_chip_aspect,
        )
        print(f"defect_report={generated_path}")
        if external_xy_csv_path:
            print(f"wafer_map={wafer_map_path}")


if __name__ == "__main__":
    main()
