#!/usr/bin/env python3
"""
AION standalone runner — incremental DSG + multi-session.

Algorithm (faithful to AION ROS node):
  Training:
    For each frame  →  pipeline.step()
                    →  aion.update_node_positions(DSG layer-20 nodes)
    For each detection at t ≤ frame_ts  →  aion.add_detection(...)
    Every time_window seconds           →  aion.flush_window(...)
      (flush checks each hash cell: if a DSG node is nearby, push counts to FreMEn)

  The model accumulates across ALL training sessions.  Hash cells are
  position-based, so they are consistent across sessions regardless of
  session-specific DSG node IDs.

  Scoring (no pipeline needed):
    aion.log_prob(t_ns, x, y, theta)  →  find nearest trained cell, return log-prob

Usage:
    python -m aion_python.runner \\
        --config       hydra_config.yaml \\
        --train-scenes scene_day1 scene_day2 scene_day3 \\
        --train-csvs   tracks_day1.csv tracks_day2.csv tracks_day3.csv \\
        --test-csvs    tracks_day4.csv [tracks_day5.csv ...] \\
        --output       preds_aion.csv \\
        [--assoc-radius 0.7] [--grid-size 0.3] [--time-window 10.0] [--order 1]
"""
import argparse
import csv
import math
import pathlib
import sys

from . import AionCore, AionConfig

LAYER_MESH_PLACES = 20


# ── DSG helpers ───────────────────────────────────────────────────────────────

def _extract_layer20(graph):
    """Return list of (node_id, x, y, z) from DSG layer 20, or [] if absent."""
    if not graph.has_layer(LAYER_MESH_PLACES):
        return []
    layer = graph.get_layer(LAYER_MESH_PLACES)
    out = []
    for node in layer.nodes:
        pos = graph.get_position(node.id)
        out.append((int(node.id.value), float(pos[0]), float(pos[1]), float(pos[2])))
    return out


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_detections(csv_path):
    """Load a tracks CSV → list of (t_ns, x, y, theta) sorted by t_ns."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((
                int(row["t_ns"]),
                float(row["x_world"]),
                float(row["y_world"]),
                float(row["theta_rad"]),
            ))
    rows.sort(key=lambda r: r[0])
    return rows


# ── Training ──────────────────────────────────────────────────────────────────

def _train_session(pipeline, aion, detections, time_window_ns, scene_path):
    """
    Drive one scene through the Hydra pipeline.

    After each frame:
      - update hash-cell DSG node positions
      - feed detections whose timestamp ≤ current frame timestamp
      - flush FreMEn window when the window boundary is crossed
    """
    try:
        from hydra_python.data_loader import FileDataLoader, DataLoaderIter
    except ImportError:
        sys.exit(
            "hydra_python not found.\n"
            "Install it with:\n"
            "  source <aion_ws>/devel/setup.bash\n"
            "  cd <aion_ws>/src/hydra/python && pip install -e ."
        )

    loader = FileDataLoader(scene_path)
    det_idx = 0
    next_window_end_ns = None
    n_frames = 0
    n_det_fed = 0

    for packet in DataLoaderIter(loader):
        ts_ns = int(packet.timestamp)
        n_frames += 1

        # Step Hydra → build DSG incrementally
        pipeline.step(
            ts_ns,
            packet.world_t_body,
            packet.world_q_body,
            packet.depth,
            packet.labels,
            packet.color,
        )

        # Update hash-cell node-position cache from current DSG
        aion.update_node_positions(_extract_layer20(pipeline.graph))

        # Feed all detections up to this frame's timestamp
        while det_idx < len(detections) and detections[det_idx][0] <= ts_ns:
            t_ns, x, y, theta = detections[det_idx]
            det_idx += 1

            # Initialise window on first detection
            if next_window_end_ns is None:
                next_window_end_ns = t_ns + time_window_ns

            # Flush any completed windows before adding this detection
            while t_ns >= next_window_end_ns:
                aion.flush_window(next_window_end_ns)
                next_window_end_ns += time_window_ns

            aion.add_detection(t_ns, x, y, theta)
            n_det_fed += 1

    # Flush the last partial window
    if next_window_end_ns is not None:
        aion.flush_window(next_window_end_ns)

    name = pathlib.Path(scene_path).name
    print(
        f"  {name}: {n_frames} frames, "
        f"{n_det_fed}/{len(detections)} detections fed, "
        f"{aion.num_node_entries} node entries / "
        f"{aion.num_hash_cells} unbound hash cells"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run(
    config_path,
    train_scenes,
    train_csvs,
    test_csvs,
    output_csv,
    assoc_radius=0.7,
    grid_size=0.3,
    time_window_s=10.0,
    fremen_order=1,
    candidate_periods=None,
):
    try:
        import hydra_python as hydra
        from hydra_python.data_loader import FileDataLoader
    except ImportError:
        sys.exit(
            "hydra_python not found.\n"
            "Install it with:\n"
            "  source <aion_ws>/devel/setup.bash\n"
            "  cd <aion_ws>/src/hydra/python && pip install -e ."
        )

    cfg = AionConfig()
    cfg.assoc_radius = assoc_radius
    cfg.grid_size    = grid_size
    cfg.fremen_order = fremen_order
    if candidate_periods:
        cfg.candidate_periods = list(candidate_periods)

    model = AionCore(cfg)
    time_window_ns = int(time_window_s * 1_000_000_000)

    # Create pipeline once using the first training scene's camera config
    sensor   = FileDataLoader(train_scenes[0]).sensor
    pipeline = hydra.HydraPipeline.from_file(str(config_path), sensor)

    # ── Training (all sessions, model accumulates across them) ────────────────
    print(f"Training on {len(train_scenes)} session(s)...")
    for i, (scene_path, csv_path) in enumerate(zip(train_scenes, train_csvs)):
        print(f"  Session {i+1}/{len(train_scenes)}: {pathlib.Path(scene_path).name}")
        detections = _load_detections(csv_path)
        # Reset Hydra map for each new session — AionCore state persists
        if i > 0:
            pipeline.reset()
        _train_session(pipeline, model, detections, time_window_ns, scene_path)

    pipeline.save()
    print(
        f"Training complete: {model.num_node_entries} node entries, "
        f"{model.num_hash_cells} unbound hash cells remaining."
    )

    # ── Scoring ───────────────────────────────────────────────────────────────
    # No pipeline needed: logProb queries hash-cell FreMEn models directly.
    print(f"Scoring {len(test_csvs)} test CSV(s)...")
    with open(output_csv, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "t_ns", "ped_id", "log_p_heading", "log_p_speed", "log_p_joint", "cell_id"
        ])
        n_scored = n_nan = 0
        for csv_path in test_csvs:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    t_ns  = int(row["t_ns"])
                    lp    = model.log_prob(
                        t_ns,
                        float(row["x_world"]),
                        float(row["y_world"]),
                        float(row["theta_rad"]),
                    )
                    if math.isnan(lp):
                        n_nan += 1
                        lp_str = ""
                    else:
                        lp_str = lp
                    writer.writerow(
                        [t_ns, row["ped_id"], lp_str, "", lp_str, ""]
                    )
                    n_scored += 1

    print(f"Scored {n_scored} detections → {output_csv}")
    if n_nan:
        print(
            f"  {n_nan} had no trained cell within {assoc_radius} m "
            f"(written as empty — skipped in evaluate.py)."
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AION standalone runner")
    p.add_argument(
        "--config", required=True,
        help="Hydra pipeline config YAML.",
    )
    p.add_argument(
        "--train-scenes", nargs="+", required=True, metavar="SCENE_DIR",
        help="FileDataLoader scene dirs for training (one per session).",
    )
    p.add_argument(
        "--train-csvs", nargs="+", required=True, metavar="CSV",
        help="Pedestrian tracks CSVs for training (same order as --train-scenes).",
    )
    p.add_argument(
        "--test-csvs", nargs="+", required=True, metavar="CSV",
        help="Pedestrian tracks CSV(s) to score.",
    )
    p.add_argument("--output", required=True, help="Output predictions CSV.")
    p.add_argument(
        "--assoc-radius", type=float, default=0.7,
        help="Max distance (m) to match a cell/detection to a DSG node "
             "(default 0.7, matches AION detection_association_distance).",
    )
    p.add_argument(
        "--grid-size", type=float, default=0.3,
        help="Spatial hash grid cell size in metres "
             "(default 0.3, matches AION spatial_hash_grid_size).",
    )
    p.add_argument(
        "--time-window", type=float, default=10.0, dest="time_window_s",
        help="FreMEn observation window duration in seconds "
             "(default 10, matches AION update_interval_seconds).",
    )
    p.add_argument(
        "--order", type=int, default=1, dest="fremen_order",
        help="FreMEn spectral order used at prediction time.",
    )
    args = p.parse_args()

    if len(args.train_scenes) != len(args.train_csvs):
        p.error("--train-scenes and --train-csvs must have the same number of entries.")

    run(
        config_path   = args.config,
        train_scenes  = args.train_scenes,
        train_csvs    = args.train_csvs,
        test_csvs     = args.test_csvs,
        output_csv    = args.output,
        assoc_radius  = args.assoc_radius,
        grid_size     = args.grid_size,
        time_window_s = args.time_window_s,
        fremen_order  = args.fremen_order,
    )


if __name__ == "__main__":
    main()
