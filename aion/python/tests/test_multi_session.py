"""
Multi-session correctness tests for AION.

The contract: after save_state -> load_state, log_prob queries at the same
(t_ns, x, y, theta) must produce identical results (the FreMEn models are
fully reconstructed by replaying the observation log, so the spectral state
is bit-identical modulo float rounding).

The split-then-resume test confirms the deeper invariant: feeding the same
detection stream in two passes (with a save/load in the middle) produces the
same FreMEn models as a single uninterrupted pass.

Requires _aion_standalone to be built and importable.
"""
import math
import os
import tempfile

import pytest

from aion_python import AionConfig, AionCore


# ── Synthetic fixtures ──────────────────────────────────────────────────────

# A tiny "scene" with three places and detections drifting through them.
_PLACES = [
    (1, 0.0, 0.0, 0.0),
    (2, 5.0, 0.0, 0.0),
    (3, 0.0, 5.0, 0.0),
]


def _make_detections(n_windows=8, det_per_window=20, window_s=10.0):
    """Generate synthetic detections at the three places, varying heading."""
    out = []
    base_t = 1_000_000_000  # 1s in ns, avoid t=0 edge cases
    for w in range(n_windows):
        t_base = base_t + int(w * window_s * 1e9)
        for k in range(det_per_window):
            t = t_base + int((k / det_per_window) * window_s * 1e9)
            # Cycle through the three places + bias heading per window
            px, py = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)][k % 3]
            theta = (w * 0.3 + k * 0.05) % (2 * math.pi)
            out.append((t, px, py, theta))
    return out


def _make_core():
    cfg = AionConfig()
    cfg.assoc_radius = 0.7
    cfg.grid_size = 0.3
    cfg.fremen_order = 1
    return AionCore(cfg)


# ── Tests ───────────────────────────────────────────────────────────────────

def test_save_load_roundtrip_preserves_log_prob():
    """save_state -> load_state must preserve log_prob queries exactly."""
    core_a = _make_core()
    core_a.update_node_positions(_PLACES)
    dets = _make_detections()
    window_ns = 10_000_000_000
    next_window_end = dets[0][0] + window_ns
    for t, x, y, theta in dets:
        while t >= next_window_end:
            core_a.flush_window(next_window_end)
            next_window_end += window_ns
        core_a.add_detection(t, x, y, theta)
    core_a.flush_window(next_window_end)

    queries = [
        (dets[-1][0], 0.0, 0.0, 0.5),
        (dets[-1][0], 5.0, 0.0, 1.2),
        (dets[-1][0], 0.0, 5.0, 3.0),
    ]
    expected = [core_a.log_prob(*q) for q in queries]
    # Sanity: at least one query should hit a trained cell (not NaN).
    assert any(not math.isnan(v) for v in expected), \
        "fixture produced no trained cells — adjust _make_detections"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "aion_state.bin")
        ok, err = core_a.save_state(path)
        assert ok, f"save_state failed: {err}"
        assert os.path.getsize(path) > 0

        core_b = _make_core()
        # The new core needs the DSG positions cached so log_prob can find
        # the nearest trained node entry (entries are keyed by node-grid
        # position, the cache is rebuilt fresh each session).
        core_b.update_node_positions(_PLACES)
        ok, err = core_b.load_state(path)
        assert ok, f"load_state failed: {err}"
        assert core_b.num_node_entries == core_a.num_node_entries

    actual = [core_b.log_prob(*q) for q in queries]
    for a, b, q in zip(expected, actual, queries):
        if math.isnan(a):
            assert math.isnan(b), f"query {q}: NaN -> {b}"
        else:
            assert math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6), \
                f"query {q}: {a} != {b}"


def test_load_rejects_grid_size_mismatch():
    """Loading a state file into a core with a different grid_size must fail."""
    core_a = _make_core()
    core_a.update_node_positions(_PLACES)
    for t, x, y, theta in _make_detections(n_windows=2):
        core_a.add_detection(t, x, y, theta)
    core_a.flush_window(_make_detections(n_windows=2)[-1][0] + 1_000_000_000)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "aion_state.bin")
        ok, _ = core_a.save_state(path)
        assert ok

        cfg = AionConfig()
        cfg.grid_size = 0.5  # mismatched
        core_b = AionCore(cfg)
        ok, err = core_b.load_state(path)
        assert not ok, "load_state should reject mismatched grid_size"
        assert "grid_size" in err.lower()


def test_load_rejects_missing_file():
    core = _make_core()
    ok, err = core.load_state("/nonexistent/path/aion_state.bin")
    assert not ok
    assert err


def test_split_then_resume_matches_single_shot():
    """Feeding detections in two passes (with save+load mid-stream) must
    produce the same trained state as a single uninterrupted pass."""
    dets = _make_detections(n_windows=6)
    window_ns = 10_000_000_000
    mid = len(dets) // 2
    mid_t = dets[mid][0]

    # ── Single-shot reference ──
    ref = _make_core()
    ref.update_node_positions(_PLACES)
    next_w = dets[0][0] + window_ns
    for t, x, y, theta in dets:
        while t >= next_w:
            ref.flush_window(next_w)
            next_w += window_ns
        ref.add_detection(t, x, y, theta)
    ref.flush_window(next_w)

    # ── Split-then-resume ──
    a = _make_core()
    a.update_node_positions(_PLACES)
    next_w = dets[0][0] + window_ns
    for t, x, y, theta in dets[:mid]:
        while t >= next_w:
            a.flush_window(next_w)
            next_w += window_ns
        a.add_detection(t, x, y, theta)
    a.flush_window(next_w)  # close out the last partial window before saving

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "aion_state.bin")
        ok, err = a.save_state(path)
        assert ok, err

        b = _make_core()
        ok, err = b.load_state(path)
        assert ok, err
        # load_state resets the transient DSG node cache; re-supply positions
        # (as the runner does every frame) so the resumed second half binds to
        # nodes and merges into the loaded first-half history rather than
        # fragmenting into a separate hash cell.
        b.update_node_positions(_PLACES)

        # Resume — pick up the window timer from where the first half left off.
        next_w_b = next_w
        for t, x, y, theta in dets[mid:]:
            while t >= next_w_b:
                b.flush_window(next_w_b)
                next_w_b += window_ns
            b.add_detection(t, x, y, theta)
        b.flush_window(next_w_b)

    # Compare log_prob across a grid of queries.
    queries = [
        (dets[-1][0] + 1_000_000_000, px, py, theta)
        for (_, px, py, _) in _PLACES
        for theta in (0.25, 1.0, 2.5, 4.2)
    ]
    for q in queries:
        r = ref.log_prob(*q)
        s = b.log_prob(*q)
        if math.isnan(r):
            assert math.isnan(s), f"query {q}: ref NaN, resume {s}"
        else:
            # The replay is bit-identical at the observation level; only
            # float ordering can drift. Tolerance per the plan: 1e-2.
            assert not math.isnan(s), f"query {q}: ref={r}, resume NaN"
            assert math.isclose(r, s, rel_tol=1e-2, abs_tol=1e-3), \
                f"query {q}: ref={r} != resume={s}"
