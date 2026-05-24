#pragma once
#include <libfremen/fremen_model.hpp>
#include <Eigen/Core>
#include <array>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>

namespace aion {

static constexpr int kNBins = 8;

struct AionConfig {
    // Max XY distance (m) to associate a hash cell to a DSG node at bind time,
    // and to find the nearest place at log_prob query time.
    // Matches AION detection_association_distance.
    double assoc_radius = 0.7;
    // Spatial hash grid cell size (m). Matches AION spatial_hash_grid_size.
    double grid_size = 0.3;
    int    fremen_order = 1;
    std::vector<float> candidate_periods;
};

// ---------------------------------------------------------------------------
// AionCore — offline replication of TemporalDynamicsNode's algorithm.
//
// Storage mirrors AION's two-tier design:
//
//   hash_cells_      Temporary accumulator for unbound positions.
//                    Window counts only; no FreMEn model.
//                    Keyed by grid cell of the DETECTION position.
//
//   node_entries_    Permanent FreMEn models, created when a hash cell is first
//                    bound to a DSG node (flushWindow).
//                    Keyed by the grid cell of the NODE position — position-
//                    based so the same physical place gets the same key across
//                    sessions even when DSG node IDs differ.
//
//   hash_bindings_   Maps cell_key → node_entry_key so future detections at an
//                    already-bound position bypass the hash cell and go straight
//                    to the node entry.  Persists across session resets.
//
// Flow (per time window):
//   addDetection  →  hash cell (unbound) or node entry (already bound)
//   flushWindow   →  try to bind unbound cells to nearest DSG node;
//                    push normalised counts to the node entry's FreMEn model;
//                    keep unbound cells whose counts cannot be flushed yet
//   logProb       →  find nearest node entry within assoc_radius; query FreMEn
// ---------------------------------------------------------------------------
class AionCore {
public:
    explicit AionCore(AionConfig cfg = {});

    // Replace the DSG node position cache with the current pipeline.graph
    // snapshot (layer 20).  Call after every pipeline.step() during training.
    // Only positions are used; node IDs are NOT the long-term FreMEn key.
    void updateNodePositions(
        const std::vector<std::tuple<uint64_t, double, double, double>>& nodes);

    // Accumulate one pedestrian detection into the appropriate hash cell or
    // node entry (whichever the position has been bound to).
    void addDetection(int64_t t_ns, double x, double y, double theta);

    // End-of-window step:
    //   1. Flush node entries that have accumulated counts → FreMEn.
    //   2. Try to bind each unbound hash cell to the nearest DSG node:
    //        - If a node is found: create/update the node entry, push the
    //          accumulated counts, record the binding, clear the hash cell.
    //        - If no node is found: retain counts for the next window
    //          (same behaviour as AION for detections in unmapped areas).
    void flushWindow(int64_t window_end_ns);

    // Predict log P(bin(theta) | nearest_node_entry, t).
    // Returns NaN if no trained node entry is within assoc_radius.
    double logProb(int64_t t_ns, double x, double y, double theta) const;

    void reset();

    std::size_t numHashCells()    const { return hash_cells_.size(); }
    std::size_t numNodeEntries()  const { return node_entries_.size(); }
    // Entries with at least one FreMEn observation (all node entries qualify
    // once they have been flushed at least once).
    std::size_t numTrainedEntries() const { return node_entries_.size(); }

private:
    // Temporary accumulator — no FreMEn model.
    struct HashCell {
        int64_t ix = 0, iy = 0;
        std::array<int, kNBins> window_counts{};
        int window_total = 0;
    };

    // Permanent FreMEn store — one per physical place.
    struct NodeEntry {
        Eigen::Vector3d pos;                                    // node position
        std::array<libfremen::FremenCellModel, kNBins> models;
        std::array<int, kNBins> window_counts{};
        int window_total = 0;
        bool has_data = false;  // true after first FreMEn observation
    };

    AionConfig config_;

    // Current DSG nodes (replaced by updateNodePositions after each frame).
    std::unordered_map<uint64_t, Eigen::Vector3d> nodes_cache_;

    std::unordered_map<size_t, HashCell>   hash_cells_;    // cell_key → cell
    std::unordered_map<size_t, NodeEntry>  node_entries_;  // node_key → entry
    std::unordered_map<size_t, size_t>     hash_bindings_; // cell_key → node_key

    std::vector<float> periods_;

    // Grid helpers
    static size_t   gridKey(int64_t ix, int64_t iy);
    void            cellIndices(double x, double y, int64_t& ix, int64_t& iy) const;
    std::pair<double,double> cellCenter(int64_t ix, int64_t iy) const;

    // Find nearest DSG node to (cx, cy) within assoc_radius; return its
    // position-based grid key, or nullopt.
    std::optional<std::pair<size_t, Eigen::Vector3d>>
        findNearestNode(double cx, double cy) const;

    // Find nearest node entry with has_data==true to (x, y) within assoc_radius.
    std::optional<size_t> findNearestNodeEntry(double x, double y) const;

    void initModels(NodeEntry& ne) const;
    int  thetaToBin(double theta) const;
};

}  // namespace aion
