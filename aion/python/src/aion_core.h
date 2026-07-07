#pragma once
#include <libfremen/fremen_model.hpp>
#include <Eigen/Core>
#include <array>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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
    // Spatial hash grid cell size (m).
    double grid_size = 0.4;
    int    fremen_order = 1;
    // Empty → libfremen daily/weekly periods. The runner overrides this with
    // [60, 300, 600] s (minute-scale), the correct scale for the dataset.
    std::vector<float> candidate_periods;
    // Detection range gate: drop detections farther than this (metres) from the
    // robot pose. Limits the local support window.
    // Default infinity = gate disabled.
    double max_detection_range_m = std::numeric_limits<double>::infinity();
};

// ---------------------------------------------------------------------------
// AionCore — offline replication of TemporalDynamicsNode's algorithm.
//
// Temporal data is anchored to STABLE spatial grid cells; a DSG node binding is
// a transient overlay, mirroring AION's ROS temporal_place_data_ + bind/transfer
// + handleNodeRemoval. There is ONE conceptual global FreMEn model: each grid
// cell × orientation bin is an independent FreMEn element (FremenArray elements
// do not couple), so per-element models are mathematically equivalent to a
// single concatenated global array.
//
//   hash_cells_      Unbound places, keyed by the DETECTION position's grid cell.
//   node_entries_    Bound places, keyed by the NODE position's grid cell
//                    (position-based so the same physical place gets the same
//                    key across sessions even when DSG node IDs differ).
//   hash_bindings_   detection cell_key → node_entry_key; routes post-binding
//                    detections straight to the node entry.
//
// Each Entry stores the RAW per-window count history (t_s → counts[kNBins]) and
// builds its FreMEn models lazily (the models are a pure function of history).
// Storing raw counts — rather than one addObservation() per cell per window —
// lets multiple cells merge into one node (and a node release back to a cell)
// by summing counts at each t_s, and avoids the duplicate-timestamp loss that
// per-observation feeding suffers when several cells bind to one node in the
// same window.
//
// Flow (per time window):
//   addDetection  →  bound node entry (via hash_bindings_) or hash cell
//   flushWindow   →  1. commit each entry's current window into its history
//                    2. releaseRemovedNodes(): move vanished bound nodes back
//                       to hash space (merge by t_s)
//                    3. bind unbound hash cells to the nearest live node
//                       (merge by t_s), record the binding, erase the cell
//   logProb       →  nearest entry (either map) with history within assoc_radius;
//                    build its models lazily; predict.
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
    // robot_x/robot_y: world-frame robot position used for the range gate. When
    // the detection is farther than config.max_detection_range_m from the robot
    // it is dropped. Defaults (0,0) + infinite range == no gating.
    void addDetection(int64_t t_ns, double x, double y, double theta,
                      double robot_x = 0.0, double robot_y = 0.0);

    // End-of-window step: commit windows → history, release vanished nodes back
    // to hash space, then bind unbound hash cells to the nearest DSG node.
    void flushWindow(int64_t window_end_ns);

    // Predict log P(bin(theta) | nearest place, t).
    // Returns NaN if no place with history is within assoc_radius.
    double logProb(int64_t t_ns, double x, double y, double theta) const;

    // Heading log-DENSITY log p(theta) for the channel-wise MLPD table. Aion is
    // heading-only, so this is its only density channel (speed/joint are "--").
    // Converts the per-bin probability mass P(bin) to a density over [0, 2pi):
    // p(theta) = P(bin) * kNBins / (2pi), i.e. logProb + log(kNBins/(2pi)).
    // NaN if no trained cell within assoc_radius.
    double logProbHeading(int64_t t_ns, double x, double y, double theta) const;

    // Full per-bin heading distribution at the query point and time: the
    // normalized bin masses P(bin | nearest entry, t) in bin index order.
    // Empty if no trained entry lies within assoc_radius. One entry lookup
    // and one model build serve every bin, so callers that need the whole
    // distribution should prefer this over kNBins single-bin queries.
    std::vector<double> headingDistribution(int64_t t_ns, double x, double y) const;

    // Bin index for a heading: the same mapping used to accumulate and score.
    int thetaToBin(double theta) const;

    void reset();

    std::size_t numHashCells()    const { return hash_cells_.size(); }
    std::size_t numNodeEntries()  const { return node_entries_.size(); }
    std::size_t numTrainedEntries() const;

    // ── Multi-session state I/O ──────────────────────────────────────────────
    // Custom binary format (magic "AION" + version). Stores the raw per-window
    // count history of every entry; FreMEn models are rebuilt lazily on first
    // query, so the format is independent of libfremen's serialization.
    // grid_size + n_bins must match on load. hash_bindings_ is not persisted
    // (next flushWindow re-binds to the new DSG run's places).
    bool saveState(const std::string& path,
                   std::string* error = nullptr) const;
    bool loadState(const std::string& path,
                   std::string* error = nullptr);

private:
    // Unified spatial place entry — anchors raw temporal history + lazy models.
    struct Entry {
        int64_t ix = 0, iy = 0;
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        std::array<int, kNBins> window_counts{};
        int window_total = 0;
        // Committed per-window raw counts (t_s seconds → counts per bin).
        std::vector<std::pair<uint32_t, std::array<int, kNBins>>> history;
        // FreMEn models, lazily (re)built from history; mutable so const logProb
        // can build on demand. models_built is cleared whenever history changes.
        mutable std::array<libfremen::FremenCellModel, kNBins> models;
        mutable bool models_built = false;
        bool hasData() const { return !history.empty(); }
    };

    AionConfig config_;

    std::unordered_map<uint64_t, Eigen::Vector3d> nodes_cache_;  // current DSG nodes
    std::unordered_map<size_t, Entry> hash_cells_;     // unbound, keyed by detection cell
    std::unordered_map<size_t, Entry> node_entries_;   // bound, keyed by node cell
    std::unordered_map<size_t, size_t> hash_bindings_; // detection cell_key → node_key

    std::vector<float> periods_;

    // Grid helpers
    static size_t   gridKey(int64_t ix, int64_t iy);
    void            cellIndices(double x, double y, int64_t& ix, int64_t& iy) const;
    std::pair<double,double> cellCenter(int64_t ix, int64_t iy) const;

    // Nearest live DSG node to (cx, cy) within assoc_radius → (node cell key, pos).
    std::optional<std::pair<size_t, Eigen::Vector3d>>
        findNearestNode(double cx, double cy) const;

    // Nearest entry (either map) with history within assoc_radius, or nullptr.
    const Entry* findNearestEntry(double x, double y) const;

    void initModels(std::array<libfremen::FremenCellModel, kNBins>& models) const;
    void buildModels(const Entry& e) const;
    void commitWindow(Entry& e, uint32_t t_s) const;
    void mergeInto(Entry& dst, const Entry& src) const;
    void releaseRemovedNodes();
};

}  // namespace aion
