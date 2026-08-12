from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_profiles import (  # pyright: ignore[reportMissingImports]
    load_json_config,
    product_config_args,
    runtime_defaults,
    runtime_inference_args,
)
from .pipeline_product import run_product
from rect_detector.main_inferer import MainInferer


_REQUIRED_LIGHT_DIRS = tuple(f"Light{light}-raw" for light in range(1, 5))
_MANAGED_PRODUCT_OPTIONS = {
    "--output-dir",
    "--output-json",
    "--predict-input-root",
    "--defect-report-path",
    "--external-xy-csv-path",
    "--wafer-map-path",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path).resolve()


def _is_product_ready(product_dir: Path) -> tuple[bool, str]:
    missing_dirs = [name for name in _REQUIRED_LIGHT_DIRS if not (product_dir / name).is_dir()]
    if missing_dirs:
        return False, f"missing required folders: {', '.join(missing_dirs)}"

    if not any((product_dir / "Light3-raw").glob("IMAGE3_*.raw")):
        return False, "Light3-raw contains no IMAGE3_*.raw files"

    return True, ""


def _discover_product_dirs(imgs_root: Path, order_by: str, reverse: bool) -> list[Path]:
    products = [path for path in imgs_root.iterdir() if path.is_dir()]
    if order_by == "name":
        return sorted(products, key=lambda path: path.name, reverse=reverse)

    # Directory mtime records the order in which product folders are created by the line.
    return sorted(
        products,
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=reverse,
    )


def _write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(f"{summary_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(summary_path)


def _has_managed_option(args: list[str]) -> str | None:
    for arg in args:
        option = arg.split("=", 1)[0]
        if option in _MANAGED_PRODUCT_OPTIONS:
            return option
    return None


def _build_product_args(
    product_dir: Path,
    product_output_dir: Path,
    forwarded_args: list[str],
    defect_report: bool,
    auto_external_xy_csv: bool,
) -> list[str]:
    product_name = product_dir.name
    output_json_path = product_output_dir / "json" / f"{product_name}.json"
    predict_input_root = product_output_dir / "predict_input"
    product_args = [
        str(product_dir),
        *forwarded_args,
        "--output-dir",
        str(product_output_dir),
        "--output-json",
        str(output_json_path),
        "--predict-input-root",
        str(predict_input_root),
    ]

    if defect_report:
        product_args.extend(
            [
                "--defect-report",
                "--defect-report-path",
                str(product_output_dir / "csv" / f"{product_name}.csv"),
            ]
        )

    external_xy_csv_path = product_dir / f"{product_dir.name}.csv"
    if auto_external_xy_csv and external_xy_csv_path.is_file():
        product_args.extend(
            [
                "--external-xy-csv-path",
                str(external_xy_csv_path),
                "--wafer-map-path",
                str(product_output_dir / "plots" / f"{product_name}_wafer_map.png"),
            ]
        )

    return product_args


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    project_root = _project_root()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--product-config",
        type=Path,
        default=Path("configs/products/48AMA.json"),
    )
    config_parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/runtime/production.json"),
    )
    config_args, _ = config_parser.parse_known_args(argv)
    product_config_path, product_config = load_json_config(
        config_args.product_config, project_root, "product config"
    )
    runtime_config_path, runtime_config = load_json_config(
        config_args.runtime_config, project_root, "runtime config"
    )
    defaults = runtime_defaults(runtime_config)
    default_output_root = defaults["output_root"] or str(
        Path("outputs") / f"line_run_{datetime.now():%Y%m%d_%H%M%S}"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the product pipeline for all completed product folders under a production-line imgs directory. "
            "Any unrecognised arguments are forwarded to pipeline_product.py."
        )
    )
    parser.add_argument(
        "imgs_root",
        nargs="?",
        type=Path,
        default=Path(defaults["imgs_root"]),
        help="Production-line image root containing one directory per product.",
    )
    parser.add_argument("--product-config", type=Path, default=product_config_path)
    parser.add_argument("--runtime-config", type=Path, default=runtime_config_path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(default_output_root),
        help="Root for this line run; each product receives an isolated subdirectory.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(defaults["summary_json"]) if defaults["summary_json"] else None,
        help="Line-run summary JSON path. Defaults to OUTPUT_ROOT/line_summary.json.",
    )
    parser.add_argument("--order-by", choices=("mtime", "name"), default=defaults["order_by"])
    parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction,
        default=defaults["reverse"], help="Process newest product first."
    )
    parser.add_argument(
        "--max-products", type=int, default=defaults["max_products"],
        help="Process at most N products (0 means all)."
    )
    parser.add_argument(
        "--skip-incomplete",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_incomplete"],
        help="Skip folders that are still being written by the production line.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_completed"],
        help="Skip products with an existing per-product result JSON in OUTPUT_ROOT.",
    )
    parser.add_argument(
        "--fail-fast", action=argparse.BooleanOptionalAction,
        default=defaults["fail_fast"], help="Stop at the first failed product."
    )
    parser.add_argument(
        "--watch",
        action=argparse.BooleanOptionalAction,
        default=defaults["watch"],
        help="Continuously rescan for newly completed product folders.",
    )
    parser.add_argument(
        "--rescan-interval",
        type=float,
        default=defaults["rescan_interval"],
        help="Seconds between scans in --watch mode. Default: 30.",
    )
    parser.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction,
        default=defaults["dry_run"], help="List selected products without running inference."
    )
    parser.add_argument(
        "--defect-report",
        action=argparse.BooleanOptionalAction,
        default=defaults["defect_report"],
        help="Write one defect-report CSV per product. Default: enabled.",
    )
    parser.add_argument(
        "--auto-external-xy-csv",
        action=argparse.BooleanOptionalAction,
        default=defaults["auto_external_xy_csv"],
        help="Use PRODUCT_DIR/PRODUCT_NAME.csv when present and generate its wafer map. Default: enabled.",
    )
    parser.add_argument(
        "--save-predict-input",
        action=argparse.BooleanOptionalAction,
        default=defaults["save_predict_input"],
        help="Save model input crops for every product. Default: disabled.",
    )
    parser.add_argument(
        "--save-predict-input-only-with-boxes",
        action=argparse.BooleanOptionalAction,
        default=defaults["save_predict_input_only_with_boxes"],
        help="When saving crops, save only crops where YOLO detected at least one box.",
    )
    args, cli_forwarded_args = parser.parse_known_args(argv)

    forwarded_args = [
        *runtime_inference_args(runtime_config),
        *product_config_args(product_config, project_root),
        *cli_forwarded_args,
    ]
    forwarded_args.append(
        "--save-predict-input" if args.save_predict_input else "--no-save-predict-input"
    )
    forwarded_args.append(
        "--save-predict-input-only-with-boxes"
        if args.save_predict_input_only_with_boxes
        else "--no-save-predict-input-only-with-boxes"
    )

    managed_option = _has_managed_option(forwarded_args)
    if managed_option:
        parser.error(
            f"{managed_option} is managed per product by this line runner; use --output-root instead."
        )
    if args.max_products < 0:
        parser.error("--max-products must be >= 0")
    if args.rescan_interval <= 0:
        parser.error("--rescan-interval must be > 0")
    return args, forwarded_args


def main(argv: list[str] | None = None) -> None:
    args, forwarded_args = _parse_args(argv)
    project_root = _project_root()
    imgs_root = _resolve_from_project(args.imgs_root, project_root)
    output_root = _resolve_from_project(args.output_root, project_root)
    summary_path = _resolve_from_project(args.summary_json, project_root) if args.summary_json else output_root / "line_summary.json"

    if not imgs_root.is_dir():
        raise FileNotFoundError(f"Production-line imgs directory does not exist: {imgs_root}")

    summary: dict[str, Any] = {
        "imgs_root": str(imgs_root),
        "output_root": str(output_root),
        "product_config": str(args.product_config),
        "runtime_config": str(args.runtime_config),
        "order_by": args.order_by,
        "reverse": args.reverse,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "products": {},
    }
    processed_count = 0
    shared_inferer: MainInferer | None = None

    while True:
        product_dirs = _discover_product_dirs(imgs_root, args.order_by, args.reverse)
        found_new_product = False

        for product_dir in product_dirs:
            if args.max_products and processed_count >= args.max_products:
                break

            product_name = product_dir.name
            product_output_dir = output_root / "products" / product_name
            result_json_path = product_output_dir / "json" / f"{product_name}.json"
            existing_record = summary["products"].get(product_name, {})

            if args.skip_completed and result_json_path.is_file():
                if existing_record.get("status") != "completed":
                    summary["products"][product_name] = {
                        "status": "completed",
                        "product_dir": str(product_dir),
                        "result_json": str(result_json_path),
                        "message": "Skipped because a result JSON already exists.",
                    }
                    _write_summary(summary_path, summary)
                continue

            ready, reason = _is_product_ready(product_dir)
            if not ready:
                if not args.skip_incomplete:
                    raise RuntimeError(f"Product {product_name} is not ready: {reason}")
                if existing_record.get("status") != "incomplete" or existing_record.get("message") != reason:
                    print(f"skip incomplete product={product_name}: {reason}")
                    summary["products"][product_name] = {
                        "status": "incomplete",
                        "product_dir": str(product_dir),
                        "message": reason,
                    }
                    _write_summary(summary_path, summary)
                continue

            found_new_product = True
            processed_count += 1
            product_args = _build_product_args(
                product_dir=product_dir,
                product_output_dir=product_output_dir,
                forwarded_args=forwarded_args,
                defect_report=args.defect_report,
                auto_external_xy_csv=args.auto_external_xy_csv,
            )
            print(f"process product={product_name} output_dir={product_output_dir}")

            if args.dry_run:
                summary["products"][product_name] = {
                    "status": "dry_run",
                    "product_dir": str(product_dir),
                    "output_dir": str(product_output_dir),
                }
                _write_summary(summary_path, summary)
                continue

            started_at = datetime.now().isoformat(timespec="seconds")
            try:
                shared_inferer = run_product(product_args, inferer=shared_inferer)
            except Exception as exc:
                summary["products"][product_name] = {
                    "status": "failed",
                    "product_dir": str(product_dir),
                    "output_dir": str(product_output_dir),
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _write_summary(summary_path, summary)
                print(f"failed product={product_name}: {type(exc).__name__}: {exc}")
                if args.fail_fast:
                    raise
                continue

            summary["products"][product_name] = {
                "status": "completed",
                "product_dir": str(product_dir),
                "output_dir": str(product_output_dir),
                "result_json": str(result_json_path),
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            _write_summary(summary_path, summary)

        if not args.watch:
            break
        if args.max_products and processed_count >= args.max_products:
            break
        if not found_new_product:
            print(f"no new completed products; rescanning in {args.rescan_interval:g}s")
        time.sleep(args.rescan_interval)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _write_summary(summary_path, summary)
    print(f"line_summary={summary_path}")


if __name__ == "__main__":
    main()
