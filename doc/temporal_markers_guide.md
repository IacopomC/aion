# Temporal Dynamics Visualization Guide

## Overview

The `aion` package provides temporal markers that encode multiple dimensions of spatio-temporal flow information in a single RViz visualization. The visualization mode is set by the `visualization_mode` parameter.

## Visualization Modes

### `"direction"` mode (default)

**Topic**: `/aion/temporal_dynamics`

One arrow per active place pointing in the dominant flow direction.

| Visual Property | Encoded Information | Interpretation |
|----------------|---------------------|----------------|
| **Color** | **Entropy** | Blue = directional flow → Red = chaotic |
| **Length** | **Activity Level** | Long = many observations |
| **Alpha** | **Confidence** | Opaque = much data, transparent = little data |
| **Orientation** | **Dominant Flow Direction** | Points toward main movement direction |

### `"entropy"` mode

One sphere per active place.

| Visual Property | Encoded Information | Interpretation |
|----------------|---------------------|----------------|
| **Color** | **Entropy** | Blue = directional flow → Red = chaotic |
| **Size** | **Activity Level** | Large = many observations |
| **Alpha** | **Confidence** | Opaque = much data |

### `"bins"` mode

Multiple arrows per active place — one per orientation bin that carries enough
probability to be meaningful. This is the richest visualization: you can see the
full probability distribution over directions.

| Visual Property | Encoded Information | Interpretation |
|----------------|---------------------|----------------|
| **Shaft length** | **Bin probability** | Longer = higher probability |
| **Shaft/head width** | **Relative weight** | Thicker = dominant direction |
| **Color** | **Entropy** (place-level) | Blue = predictable, Red = chaotic |
| **Alpha** | **Relative probability** | Opaque = dominant bin, faint = minor bin |

Bins with probability below half the uniform value (`1/(2*num_bins)`) are hidden.

A two-peaked distribution (e.g., a corridor used in both directions) shows two roughly
equal arrows pointing opposite ways. A junction shows three or four arrows at roughly
equal angles.

### Inactive places

Places in the Hydra DSG that have received no detections yet are shown as small gray
spheres regardless of the visualization mode.

---

## Configuration Parameters

```yaml
temporal_dynamics_node:
  # Select visualization mode
  visualization_mode: "direction"   # "direction", "entropy", or "bins"

  # Marker appearance
  marker_scale: 0.3                 # Base size scaling factor
  arrow_scale: 1.0                  # Arrow length scaling (direction/bins modes)
  min_observations: 1               # Minimum observations to show a place marker
  num_orientation_bins: 8           # Directional resolution (45° per bin for 8 bins)

  # Color tuning
  entropy_color_mapping:
    max_entropy_for_normalization: 2.0   # Entropy value that maps to full red
```

---

## Marker Interpretation Examples

### High-traffic directional area (corridor)
- **direction mode**: Long, blue, opaque arrow
- **bins mode**: One dominant long arrow, possibly a short reverse arrow
- **Meaning**: Many observations, strong directional preference

### Low-traffic chaotic area (open plaza)
- **direction mode**: Short, red, transparent arrow
- **bins mode**: Several roughly equal short arrows pointing in different directions
- **Meaning**: Few observations, scattered directions

### Bidirectional corridor
- **direction mode**: Arrow pointing in the slightly dominant direction
- **bins mode**: Two opposite arrows of similar length — reveals the bimodality
- **Meaning**: People pass in both directions (bins mode is most informative here)

### Place without temporal data
- **All modes**: Tiny gray sphere
- **Meaning**: Hydra place exists but no detection has reached it

---

## Parameter Tuning

### For dense environments
```yaml
marker_scale: 0.2
min_observations: 5
```

### For direction-critical applications
```yaml
visualization_mode: "direction"
arrow_scale: 1.5
num_orientation_bins: 16   # Higher resolution
```

### For exploring multi-modal distributions
```yaml
visualization_mode: "bins"
arrow_scale: 1.0
num_orientation_bins: 8
```
