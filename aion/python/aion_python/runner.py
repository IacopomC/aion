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
    python -m aion_python.runner \
        --config       /hydra/datasets/dataset_config.yaml \
        --labelspace ade20k_outdoor \
        --train-scenes scene_day1 scene_day2 scene_day3 \
        --train-csvs   tracks_day1.csv tracks_day2.csv tracks_day3.csv \
        --test-csvs    tracks_day4.csv [tracks_day5.csv ...] \
        --output       outputs/preds_aion.csv \
        --output-graph  outputs/dsg.json \
        [--assoc-radius 0.7] [--grid-size 0.3] [--time-window 10.0] [--order 1]
"""
import argparse
import csv
import faulthandler
import math
import pathlib
import sys
import time

import yaml

faulthandler.enable(file=sys.stderr)

from . import AionCore, AionConfig

LAYER_MESH_PLACES = 20   # Hydra layer-20 (MESH_PLACES): the place nodes AION binds to

# Single source of truth for model params: the deployed ROS config. The binding
# reads it so the offline baseline cannot drift from the node. CLI flags override.
DEFAULT_CONFIG_YAML = (
    pathlib.Path(__file__).resolve().parents[2] / "config" / "temporal_dynamics_config.yaml"
)


def load_model_params(path):
    """Read the temporal_dynamics_node block from the ROS yaml. Returns {} if absent."""
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    return doc.get("temporal_dynamics_node", doc) or {}


def _vmrss_kb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except Exception:
        pass
    return -1


# ── DSG helpers ─────────────────────────────────────────────────────────────────

def _layer20_nodes(pipeline, layer_id=LAYER_MESH_PLACES):
    """(id, x, y, z) for every layer-`layer_id` node in the live DSG.

    Reads the current ``pipeline.graph`` snapshot (see aion_core.h: "the current
    pipeline.graph snapshot (layer 20)"); there is no ``pipeline.get_layer20_nodes()``
    method on HydraPipeline.
    """
    G = pipeline.graph
    if not G.has_layer(layer_id):
        return []
    layer = G.get_layer(layer_id)
    return [(int(n.id.value), *G.get_position(n.id.value)) for n in layer.nodes]


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

def _train_session(pipeline, aion, detections, time_window_ns, scene_path,
                   node_update_interval=10, max_steps=None,
                   mem_profile_writer=None, mem_profile_every=500):
    """
    Drive one scene through the Hydra pipeline.

    After each frame:
      - update hash-cell DSG node positions (every node_update_interval frames)
      - feed detections whose timestamp ≤ current frame timestamp
      - flush FreMEn window when the window boundary is crossed
    """
    import time
    import tqdm
    from hydra_python.data_loader import FileDataLoader, DataLoaderIter

    loader = FileDataLoader(scene_path)
    det_idx = 0
    next_window_end_ns = None
    n_frames = 0
    n_det_fed = 0
    update_ns = 0  # cumulative add_detection wall-clock (per-detection model-update)
    total = loader.num_frames if hasattr(loader, "num_frames") else None

    for step_idx, packet in enumerate(tqdm.tqdm(DataLoaderIter(loader), total=total, unit="frame", leave=False)):
        if max_steps and step_idx >= max_steps:
            break
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

        if mem_profile_writer is not None and n_frames % mem_profile_every == 0:
            G = pipeline.graph
            l3  = sum(1 for _ in G.get_layer(3).nodes)  if G.has_layer(3)  else 0
            l20 = sum(1 for _ in G.get_layer(20).nodes) if G.has_layer(20) else 0
            mem_profile_writer.writerow([n_frames, ts_ns, _vmrss_kb(), l3, l20])

        # Update hash-cell node-position cache (throttled: DSG grows slowly)
        if n_frames % node_update_interval == 0:
            aion.update_node_positions(_layer20_nodes(pipeline))

        # Robot world position for this frame (gates detections by range).
        robot_x = float(packet.world_t_body[0])
        robot_y = float(packet.world_t_body[1])

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

            _t0 = time.perf_counter_ns()
            aion.add_detection(t_ns, x, y, theta, robot_x, robot_y)
            update_ns += time.perf_counter_ns() - _t0
            n_det_fed += 1

    # Final node-position sync before flushing the last partial window
    aion.update_node_positions(_layer20_nodes(pipeline))

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
    return n_det_fed, update_ns


def _write_run_stats(stats_path, label, n_updates, update_ns_total, n_scored,
                     elapsed_s=None):
    """Write a run_stats_v1 sidecar: per-detection model-update time (ms) + memory."""
    import json
    import time
    mean_ms = (update_ns_total / n_updates) / 1e6 if n_updates else None
    rss = vms = peak_rss = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmSize:"):
                    vms = int(line.split()[1]) * 1024
                if rss is not None and vms is not None:
                    break
    except Exception:
        pass
    try:
        import resource
        mr = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_rss = mr * 1024 if mr < 2**32 else mr  # ru_maxrss is KB on Linux
    except Exception:
        pass
    rec = {
        "schema": "run_stats_v1",
        "timestamp": time.time(),
        "label": label,
        "elapsed_s": elapsed_s,
        "memory": {"rss_bytes": rss, "vms_bytes": vms, "peak_rss_bytes": peak_rss},
        "flow_timing": {
            "add_observation_count": n_updates,
            "add_observation_mean_ms": mean_ms,
            "add_observation_total_us": int(update_ns_total / 1e3),
        },
        "extra": {"n_test_detections": n_scored},
    }
    with open(stats_path, "w") as fh:
        json.dump(rec, fh, indent=2)
    if mean_ms is not None:
        print(f"[stats] per-detection model-update {mean_ms:.4f} ms x {n_updates}; "
              f"peak RSS {(peak_rss or 0)/1e6:.1f} MB -> {stats_path}")


# ── Public API ────────────────────────────────────────────────────────────────

def run(
    config_path,
    labelspace,
    train_scenes,
    train_csvs,
    test_csvs=None,
    output_csv=None,
    assoc_radius=0.7,
    grid_size=0.4,
    time_window_s=10.0,
    fremen_order=1,
    candidate_periods=(60.0, 300.0, 600.0),
    load_flow_state=None,
    save_flow_state=None,
    output_graph=None,
    max_steps=None,
    max_detection_range_m=float("inf"),
    mem_profile_csv=None,
    mem_profile_every=500,
):
    try:
        import spark_dsg  # must be imported first to register pybind11 types
        from hydra_python.data_loader import FileDataLoader
        from hydra_python.pipeline import load_pipeline
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
    cfg.max_detection_range_m = max_detection_range_m
    if candidate_periods:
        cfg.candidate_periods = list(candidate_periods)

    model = AionCore(cfg)
    time_window_ns = int(time_window_s * 1_000_000_000)

    # Warm start from a prior session's state file. Hash cells + node entries
    # (with FreMEn observation history) are restored; bindings are discarded
    # so the new session's DSG can re-bind naturally on first flush_window.
    if load_flow_state is not None:
        ok, err = model.load_state(str(load_flow_state))
        if not ok:
            sys.exit(f"[load-flow-state] failed: {err}")
        print(
            f"[load-flow-state] restored from {load_flow_state}: "
            f"{model.num_node_entries} node entries, "
            f"{model.num_hash_cells} unbound hash cells"
        )

    # Create pipeline once using the first training scene's camera config
    data     = FileDataLoader(train_scenes[0])
    pipeline = load_pipeline(data, config_path, labelspace)

    # ── Training (all sessions, model accumulates across them) ────────────────
    print(f"Training on {len(train_scenes)} session(s)...")
    total_updates = 0
    total_update_ns = 0
    _t0_train = time.time()
    _mem_fh = _mem_writer = None
    if mem_profile_csv is not None:
        _mem_fh = open(mem_profile_csv, 'w', newline='')
        _mem_writer = csv.writer(_mem_fh)
        _mem_writer.writerow(['step', 'timestamp_ns', 'vmrss_kb', 'l3_places', 'l20_places'])
    try:
        for i, (scene_path, csv_path) in enumerate(zip(train_scenes, train_csvs)):
            print(f"  Session {i+1}/{len(train_scenes)}: {pathlib.Path(scene_path).name}")
            detections = _load_detections(csv_path)
            if i > 0:
                pipeline.reset()
            nd, un = _train_session(pipeline, model, detections, time_window_ns, scene_path,
                                     max_steps=max_steps,
                                     mem_profile_writer=_mem_writer,
                                     mem_profile_every=mem_profile_every)
            total_updates += nd
            total_update_ns += un
    finally:
        if _mem_fh is not None:
            _mem_fh.close()

    pipeline.save()
    print(
        f"Training complete: {model.num_node_entries} node entries, "
        f"{model.num_hash_cells} unbound hash cells remaining."
    )

    if output_graph is not None:
        graph_path = pathlib.Path(output_graph)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.graph.save(str(graph_path))
        print(f"DSG saved → {graph_path}")

    # Persist dynamics state so a later session can resume via --load-flow-state.
    if save_flow_state is not None:
        ok, err = model.save_state(str(save_flow_state))
        if not ok:
            sys.exit(f"[save-flow-state] failed: {err}")
        print(f"[save-flow-state] wrote {save_flow_state}")
    elif test_csvs is None:
        print(
            "WARNING: no --save-flow-state given and no --test-csvs — "
            "trained model will not be persisted. Pass --save-flow-state to keep it."
        )

    # ── Scoring (optional — omit --test-csvs to skip) ─────────────────────────
    if test_csvs is None or output_csv is None:
        return

    # No pipeline needed: logProb queries hash-cell FreMEn models directly.
    print(f"Scoring {len(test_csvs)} test CSV(s)...")
    with open(output_csv, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        # evaluate_channels.py schema. Aion is heading-only -> speed/joint/mean
        # are blank (rendered "--"); log_p_heading is a DENSITY (bin mass / Δθ).
        writer.writerow([
            "t_ns", "ped_id", "theta_obs", "rho_obs", "matched",
            "log_p_joint", "log_p_heading", "log_p_speed", "mean_rho_pred",
        ])
        n_scored = n_nan = 0
        for csv_path in test_csvs:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    t_ns  = int(row["t_ns"])
                    theta = float(row["theta_rad"])
                    rho   = float(row.get("rho_mps", row.get("rho_obs", 0.0)))
                    lp_h = model.log_prob_heading(
                        t_ns,
                        float(row["x_world"]),
                        float(row["y_world"]),
                        theta,
                    )
                    matched = 0 if math.isnan(lp_h) else 1
                    if not matched:
                        n_nan += 1
                    writer.writerow([
                        t_ns, row["ped_id"], theta, rho, matched,
                        "",                            # log_p_joint   — heading-only
                        "" if not matched else lp_h,   # log_p_heading — the density
                        "",                            # log_p_speed   — no speed model
                        "",                            # mean_rho_pred — no speed model
                    ])
                    n_scored += 1

    print(f"Scored {n_scored} detections → {output_csv}")
    if n_nan:
        print(
            f"  {n_nan} had no trained cell within {assoc_radius} m "
            f"(written as empty — skipped in evaluate.py)."
        )

    _elapsed_s = time.time() - _t0_train
    _write_run_stats(str(output_csv) + ".stats.json", "aion",
                     total_updates, total_update_ns, n_scored, elapsed_s=_elapsed_s)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AION standalone runner")
    p.add_argument(
        "--config", required=True,
        help="Hydra dataset config name (e.g. 'unizar').",
    )
    p.add_argument(
        "--labelspace", default="ade20k_outdoor",
        help="Hydra label space name (default: ade20k_outdoor).",
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
        "--test-csvs", nargs="+", default=None, metavar="CSV",
        help="Pedestrian tracks CSV(s) to score. Omit to train only (use --save-flow-state to persist the model).",
    )
    p.add_argument("--output", default=None, help="Output predictions CSV. Required when --test-csvs is given.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Limit frames processed per training scene (smoke/debug; default = all).")
    p.add_argument(
        "--output-graph", default=None, metavar="PATH",
        help="Save final DSG to this path (e.g. results/dsg.json) after training.",
    )
    p.add_argument(
        "--model-config", default=str(DEFAULT_CONFIG_YAML), metavar="YAML",
        help="ROS temporal_dynamics_config.yaml — single source of truth for the "
             "model params (assoc/grid/order/periods/window). CLI flags below override it.",
    )
    p.add_argument(
        "--assoc-radius", type=float, default=None,
        help="Override detection_association_distance from the model config (m).",
    )
    p.add_argument(
        "--grid-size", type=float, default=None,
        help="Override spatial_hash_grid_size from the model config (m).",
    )
    p.add_argument(
        "--time-window", type=float, default=None, dest="time_window_s",
        help="Override update_interval_seconds (FreMEn obs window, s).",
    )
    p.add_argument(
        "--order", type=int, default=None, dest="fremen_order",
        help="Override fremen_model_order.",
    )
    p.add_argument(
        "--candidate-periods", type=float, nargs="+", default=None, metavar="SECONDS",
        help="Override candidate_periods (s).",
    )
    p.add_argument(
        "--load-flow-state", default=None, metavar="PATH",
        help="Resume from a prior session's aion_state.bin (hash cells + "
             "node entries with FreMEn history). Asserts grid_size match.",
    )
    p.add_argument(
        "--save-flow-state", default=None, metavar="PATH",
        help="Write aion_state.bin after training so a later session can "
             "resume via --load-flow-state. Skipped if not set.",
    )
    p.add_argument(
        "--max-detection-range-m", type=float, default=None,
        help="Override max_detection_range_m (m): drop detections farther than this "
             "from the robot pose. Default (yaml) = inf = disabled.",
    )
    p.add_argument(
        "--mem-profile-csv", default=None, metavar="PATH",
        help="Write per-step RSS + DSG layer counts to this CSV.",
    )
    p.add_argument(
        "--mem-profile-every", type=int, default=500, metavar="N",
        help="Sample memory every N frames (default: 500).",
    )
    args = p.parse_args()

    if len(args.train_scenes) != len(args.train_csvs):
        p.error("--train-scenes and --train-csvs must have the same number of entries.")
    if args.test_csvs is not None and args.output is None:
        p.error("--output is required when --test-csvs is given.")

    # Single source of truth: read model params from the ROS yaml; CLI flags override.
    params = load_model_params(args.model_config)

    def pick(cli, key, fallback):
        return cli if cli is not None else params.get(key, fallback)

    assoc_radius      = pick(args.assoc_radius,      "detection_association_distance", 0.7)
    grid_size         = pick(args.grid_size,         "spatial_hash_grid_size",         0.4)
    time_window_s     = pick(args.time_window_s,     "update_interval_seconds",        10.0)
    fremen_order      = pick(args.fremen_order,      "fremen_model_order",             1)
    candidate_periods = pick(args.candidate_periods, "candidate_periods",              [60.0, 300.0, 600.0])
    max_detection_range_m = pick(args.max_detection_range_m, "max_detection_range_m",  float("inf"))

    # num_orientation_bins is compiled into the binding (kNBins=8); warn on mismatch.
    nb = params.get("num_orientation_bins")
    if nb is not None and int(nb) != 8:
        print(f"[warn] config num_orientation_bins={nb} but the binding is compiled with "
              f"8 bins; using 8.", file=sys.stderr)
    print(f"[config] {args.model_config}: assoc={assoc_radius} grid={grid_size} "
          f"order={fremen_order} window={time_window_s}s periods={candidate_periods} "
          f"max_range={max_detection_range_m}")

    run(
        config_path     = args.config,
        labelspace      = args.labelspace,
        train_scenes    = args.train_scenes,
        train_csvs      = args.train_csvs,
        test_csvs       = args.test_csvs,
        output_csv      = args.output,
        assoc_radius    = assoc_radius,
        grid_size       = grid_size,
        time_window_s     = time_window_s,
        fremen_order      = fremen_order,
        candidate_periods = candidate_periods,
        load_flow_state = args.load_flow_state,
        save_flow_state = args.save_flow_state,
        output_graph    = args.output_graph,
        max_steps       = args.max_steps,
        max_detection_range_m = max_detection_range_m,
        mem_profile_csv   = args.mem_profile_csv,
        mem_profile_every = args.mem_profile_every,
    )


if __name__ == "__main__":
    main()
