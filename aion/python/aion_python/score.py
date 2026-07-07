#!/usr/bin/env python3
"""
AION standalone scorer — evaluate a saved flow-state on held-out test detections.

No pipeline or DSG needed: log_prob() queries hash-cell FreMEn models directly
from the serialized state produced by runner.py --save-flow-state.

Usage:
    python -m aion_python.score \
        --flow-state  aion_state.bin \
        --test-csvs   tracks_test1.csv [tracks_test2.csv ...] \
        --output      preds_aion.csv \
        [--assoc-radius 0.7] [--order 1]
"""
import argparse
import csv
import math
import sys

from . import AionCore, AionConfig


def score(
    flow_state: str,
    test_csvs: list[str],
    output_csv: str,
    assoc_radius: float = 0.7,
    fremen_order: int = 1,
) -> None:
    cfg = AionConfig()
    cfg.assoc_radius = assoc_radius
    cfg.fremen_order = fremen_order
    model = AionCore(cfg)

    ok, err = model.load_state(flow_state)
    if not ok:
        sys.exit(f"[flow-state] failed to load '{flow_state}': {err}")
    print(
        f"[flow-state] loaded from {flow_state}: "
        f"{model.num_node_entries} node entries, "
        f"{model.num_hash_cells} unbound hash cells"
    )

    print(f"Scoring {len(test_csvs)} test CSV(s)...")
    # Emit the preds schema consumed by the channel-wise evaluator. Aion is heading-only: it fills the heading channel
    # with a proper density p(theta) and leaves speed + joint blank, which the
    # evaluator renders as "--" (a heading-only model has no speed/joint
    # density). mean_rho_pred is blank for the same reason (speed MAE -> "--").
    with open(output_csv, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "t_ns", "ped_id", "theta_obs", "rho_obs", "matched",
            "log_p_joint", "log_p_heading", "log_p_speed", "mean_rho_pred",
        ])
        n_scored = n_nan = 0
        for csv_path in test_csvs:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    t_ns = int(row["t_ns"])
                    theta = float(row["theta_rad"])
                    rho = float(row["rho_mps"])
                    lp_h = model.log_prob_heading(
                        t_ns,
                        float(row["x_world"]),
                        float(row["y_world"]),
                        theta,
                    )
                    matched = not math.isnan(lp_h)
                    if not matched:
                        n_nan += 1
                    writer.writerow([
                        t_ns,
                        row.get("ped_id", ""),
                        theta,
                        rho,
                        int(matched),
                        "",                       # log_p_joint   — heading-only -> "--"
                        lp_h if matched else "",  # log_p_heading — the density
                        "",                       # log_p_speed   — no speed model -> "--"
                        "",                       # mean_rho_pred — no speed model
                    ])
                    n_scored += 1

    print(f"Scored {n_scored} detections → {output_csv}")
    if n_nan:
        print(
            f"  {n_nan} had no trained cell within {assoc_radius} m "
            f"(written as empty — skipped in evaluate.py)."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="AION standalone scorer (load flow-state, score test CSVs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--flow-state",   required=True,
                   help="aion_state.bin produced by runner.py --save-flow-state")
    p.add_argument("--test-csvs",    required=True, nargs="+", metavar="CSV",
                   help="Pedestrian tracks CSV(s) to score")
    p.add_argument("--output",       required=True,
                   help="Output predictions CSV path")
    p.add_argument("--assoc-radius", type=float, default=0.7,
                   help="Max XY distance (m) for nearest-cell lookup (default: 0.7)")
    p.add_argument("--order",        type=int,   default=1, dest="fremen_order",
                   help="FreMEn spectral order used at prediction time (default: 1)")
    args = p.parse_args()

    score(
        flow_state   = args.flow_state,
        test_csvs    = args.test_csvs,
        output_csv   = args.output,
        assoc_radius = args.assoc_radius,
        fremen_order = args.fremen_order,
    )


if __name__ == "__main__":
    main()
