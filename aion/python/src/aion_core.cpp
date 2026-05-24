#include "aion_core.h"
#include <algorithm>

namespace aion {

// ── Construction ─────────────────────────────────────────────────────────────

AionCore::AionCore(AionConfig cfg) : config_(std::move(cfg)) {
    periods_ = config_.candidate_periods.empty()
                   ? libfremen::defaultCandidatePeriods()
                   : config_.candidate_periods;
}

void AionCore::initModels(NodeEntry& ne) const {
    libfremen::ChangePointConfig cpc;
    libfremen::AdaptiveBasisConfig abc;
    abc.maxOrder = config_.fremen_order + 2;
    for (auto& m : ne.models) {
        m = libfremen::FremenCellModel(periods_, cpc, abc);
    }
}

// ── Grid helpers ──────────────────────────────────────────────────────────────

/*static*/ size_t AionCore::gridKey(int64_t ix, int64_t iy) {
    // Bias to handle negatives, then mix with primes to avoid collisions.
    uint64_t ux = static_cast<uint64_t>(ix + (1LL << 30));
    uint64_t uy = static_cast<uint64_t>(iy + (1LL << 30));
    return std::hash<uint64_t>()((ux * 2654435761ULL) ^ (uy * 2246822519ULL));
}

void AionCore::cellIndices(double x, double y, int64_t& ix, int64_t& iy) const {
    ix = static_cast<int64_t>(std::floor(x / config_.grid_size));
    iy = static_cast<int64_t>(std::floor(y / config_.grid_size));
}

std::pair<double, double> AionCore::cellCenter(int64_t ix, int64_t iy) const {
    return {(ix + 0.5) * config_.grid_size, (iy + 0.5) * config_.grid_size};
}

// ── DSG node cache ────────────────────────────────────────────────────────────

void AionCore::updateNodePositions(
    const std::vector<std::tuple<uint64_t, double, double, double>>& nodes) {
    nodes_cache_.clear();
    nodes_cache_.reserve(nodes.size());
    for (const auto& [id, x, y, z] : nodes) {
        nodes_cache_.emplace(id, Eigen::Vector3d(x, y, z));
    }
}

std::optional<std::pair<size_t, Eigen::Vector3d>>
AionCore::findNearestNode(double cx, double cy) const {
    size_t best_key = 0;
    Eigen::Vector3d best_pos;
    double best_d2 = config_.assoc_radius * config_.assoc_radius;
    bool found = false;

    for (const auto& [id, pos] : nodes_cache_) {
        double dx = pos.x() - cx, dy = pos.y() - cy;
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            // Key the node entry by the grid cell of the node's position so
            // the same physical place maps to the same entry across sessions
            // (node IDs differ; positions stay close).
            int64_t nix, niy;
            cellIndices(pos.x(), pos.y(), nix, niy);
            best_key = gridKey(nix, niy);
            best_pos = pos;
            found    = true;
        }
    }
    if (!found) return std::nullopt;
    return std::make_pair(best_key, best_pos);
}

// ── addDetection ──────────────────────────────────────────────────────────────

void AionCore::addDetection(int64_t /*t_ns*/, double x, double y, double theta) {
    int64_t ix, iy;
    cellIndices(x, y, ix, iy);
    size_t cell_key = gridKey(ix, iy);
    int    bin      = thetaToBin(theta);

    // If this cell has already been bound to a node entry, go there directly.
    auto bind_it = hash_bindings_.find(cell_key);
    if (bind_it != hash_bindings_.end()) {
        auto& ne = node_entries_.at(bind_it->second);
        ne.window_counts[bin]++;
        ne.window_total++;
        return;
    }

    // Otherwise accumulate in the hash cell.
    auto& cell = hash_cells_[cell_key];
    cell.ix = ix;
    cell.iy = iy;
    cell.window_counts[bin]++;
    cell.window_total++;
}

// ── flushWindow ───────────────────────────────────────────────────────────────

void AionCore::flushWindow(int64_t window_end_ns) {
    uint32_t t_s = static_cast<uint32_t>(window_end_ns / 1'000'000'000LL);

    // 1. Flush already-bound node entries (post-binding detections).
    for (auto& [node_key, ne] : node_entries_) {
        if (ne.window_total == 0) continue;
        for (int b = 0; b < kNBins; ++b) {
            float state = static_cast<float>(ne.window_counts[b]) / ne.window_total;
            ne.models[b].addObservation(t_s, state);
        }
        ne.has_data = true;
        ne.window_counts.fill(0);
        ne.window_total = 0;
    }

    // 2. Try to bind unbound hash cells.
    std::vector<size_t> to_erase;
    for (auto& [cell_key, cell] : hash_cells_) {
        if (cell.window_total == 0) continue;  // nothing to flush

        auto [cx, cy] = cellCenter(cell.ix, cell.iy);
        auto node_opt = findNearestNode(cx, cy);

        if (!node_opt) {
            // No DSG node nearby yet — retain counts for the next window.
            // This mirrors AION's behaviour in unmapped areas.
            continue;
        }

        auto [node_key, node_pos] = *node_opt;

        // Create the node entry on first binding.
        if (node_entries_.find(node_key) == node_entries_.end()) {
            NodeEntry ne;
            ne.pos = node_pos;
            initModels(ne);
            node_entries_.emplace(node_key, std::move(ne));
        }
        auto& ne = node_entries_.at(node_key);

        // Record the binding so future detections skip the hash cell.
        hash_bindings_[cell_key] = node_key;

        // Push the accumulated (pre-binding) counts to FreMEn.
        for (int b = 0; b < kNBins; ++b) {
            float state = static_cast<float>(cell.window_counts[b]) / cell.window_total;
            ne.models[b].addObservation(t_s, state);
        }
        ne.has_data = true;

        to_erase.push_back(cell_key);  // hash cell no longer needed
    }
    for (size_t k : to_erase) hash_cells_.erase(k);
}

// ── logProb ───────────────────────────────────────────────────────────────────

std::optional<size_t> AionCore::findNearestNodeEntry(double x, double y) const {
    size_t best_key = 0;
    double best_d2  = config_.assoc_radius * config_.assoc_radius;
    bool   found    = false;

    for (const auto& [node_key, ne] : node_entries_) {
        if (!ne.has_data) continue;
        double dx = ne.pos.x() - x, dy = ne.pos.y() - y;
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best_key = node_key; found = true; }
    }
    if (!found) return std::nullopt;
    return best_key;
}

double AionCore::logProb(int64_t t_ns, double x, double y, double theta) const {
    // Fast path: if this cell has been bound, go to its node entry directly.
    int64_t ix, iy;
    cellIndices(x, y, ix, iy);
    size_t cell_key = gridKey(ix, iy);

    std::optional<size_t> node_key_opt;
    auto bind_it = hash_bindings_.find(cell_key);
    if (bind_it != hash_bindings_.end()) {
        node_key_opt = bind_it->second;
    } else {
        // General search: nearest trained node entry within assoc_radius.
        node_key_opt = findNearestNodeEntry(x, y);
    }

    if (!node_key_opt) return std::numeric_limits<double>::quiet_NaN();

    const auto& ne = node_entries_.at(*node_key_opt);
    if (!ne.has_data) return std::numeric_limits<double>::quiet_NaN();

    uint32_t t_s = static_cast<uint32_t>(t_ns / 1'000'000'000LL);

    std::array<double, kNBins> probs;
    double sum = 0.0;
    for (int b = 0; b < kNBins; ++b) {
        probs[b] = std::max(1e-9,
            static_cast<double>(ne.models[b].predict(t_s, config_.fremen_order)));
        sum += probs[b];
    }
    return std::log(probs[thetaToBin(theta)] / sum);
}

// ── Utilities ─────────────────────────────────────────────────────────────────

void AionCore::reset() {
    nodes_cache_.clear();
    hash_cells_.clear();
    node_entries_.clear();
    hash_bindings_.clear();
}

int AionCore::thetaToBin(double theta) const {
    double t = std::fmod(theta, 2.0 * M_PI);
    if (t < 0) t += 2.0 * M_PI;
    return static_cast<int>(t / (2.0 * M_PI) * kNBins) % kNBins;
}

}  // namespace aion
