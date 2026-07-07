#include "aion_core.h"
#include <algorithm>
#include <cstring>
#include <fstream>

namespace aion {

namespace {
// ── Binary I/O helpers ───────────────────────────────────────────────────────
// Endianness note: benchmarks all run on x86_64 Linux. If you ever need
// cross-arch state-file portability, swap these for explicit little-endian
// encoding — for now we trust the host byte order on save and load.

template <typename T>
void writePOD(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
bool readPOD(std::istream& is, T& v) {
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return is.good();
}

void writeString(std::ostream& os, const std::string& s) {
    uint32_t n = static_cast<uint32_t>(s.size());
    writePOD(os, n);
    if (n) os.write(s.data(), n);
}

bool readString(std::istream& is, std::string& s) {
    uint32_t n = 0;
    if (!readPOD(is, n)) return false;
    s.assign(n, '\0');
    if (n) is.read(s.data(), n);
    return is.good();
}
}  // namespace

// ── Construction ─────────────────────────────────────────────────────────────

AionCore::AionCore(AionConfig cfg) : config_(std::move(cfg)) {
    periods_ = config_.candidate_periods.empty()
                   ? libfremen::defaultCandidatePeriods()
                   : config_.candidate_periods;
}

void AionCore::initModels(std::array<libfremen::FremenCellModel, kNBins>& models) const {
    libfremen::ChangePointConfig cpc;
    libfremen::AdaptiveBasisConfig abc;
    abc.maxOrder = config_.fremen_order + 2;
    for (auto& m : models) {
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
            // Key by the grid cell of the node's position so the same physical
            // place maps to the same entry across sessions (node IDs differ).
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

void AionCore::addDetection(int64_t /*t_ns*/, double x, double y, double theta,
                            double robot_x, double robot_y) {
    // Range gate: skip detections outside the robot's local support window.
    // Disabled when range == infinity.
    if (std::isfinite(config_.max_detection_range_m)) {
        const double dx = x - robot_x;
        const double dy = y - robot_y;
        if ((dx * dx + dy * dy) > (config_.max_detection_range_m * config_.max_detection_range_m)) {
            return;
        }
    }

    int64_t ix, iy;
    cellIndices(x, y, ix, iy);
    size_t cell_key = gridKey(ix, iy);
    int    bin      = thetaToBin(theta);

    // If this cell is bound to a node entry, accumulate there directly.
    auto bind_it = hash_bindings_.find(cell_key);
    if (bind_it != hash_bindings_.end()) {
        auto ne_it = node_entries_.find(bind_it->second);
        if (ne_it != node_entries_.end()) {
            ne_it->second.window_counts[bin]++;
            ne_it->second.window_total++;
            return;
        }
        // Binding points at a released node entry — drop it and fall through.
        hash_bindings_.erase(bind_it);
    }

    auto& cell = hash_cells_[cell_key];
    cell.ix = ix;
    cell.iy = iy;
    auto [cx, cy] = cellCenter(ix, iy);
    cell.pos = Eigen::Vector3d(cx, cy, 0.0);
    cell.window_counts[bin]++;
    cell.window_total++;
}

// ── Per-entry history / model helpers ──────────────────────────────────────────

void AionCore::commitWindow(Entry& e, uint32_t t_s) const {
    if (e.window_total == 0) return;
    if (!e.history.empty() && e.history.back().first == t_s) {
        for (int b = 0; b < kNBins; ++b)
            e.history.back().second[b] += e.window_counts[b];
    } else {
        e.history.emplace_back(t_s, e.window_counts);
    }
    e.window_counts.fill(0);
    e.window_total  = 0;
    e.models_built  = false;
}

void AionCore::mergeInto(Entry& dst, const Entry& src) const {
    // Combine per-window raw counts by timestamp (summing collisions), so the
    // result is the union of both observation histories at full resolution.
    std::map<uint32_t, std::array<int, kNBins>> m;
    for (const auto& [t, c] : dst.history) m[t] = c;
    for (const auto& [t, c] : src.history) {
        auto& a = m[t];  // value-initialised to zeros on first access
        for (int b = 0; b < kNBins; ++b) a[b] += c[b];
    }
    dst.history.assign(m.begin(), m.end());  // std::map keeps t_s sorted
    for (int b = 0; b < kNBins; ++b) dst.window_counts[b] += src.window_counts[b];
    dst.window_total += src.window_total;
    dst.models_built = false;
}

void AionCore::buildModels(const Entry& e) const {
    if (e.models_built) return;
    initModels(e.models);
    for (const auto& [t_s, counts] : e.history) {
        int total = 0;
        for (int b = 0; b < kNBins; ++b) total += counts[b];
        if (total <= 0) continue;
        for (int b = 0; b < kNBins; ++b)
            e.models[b].addObservation(t_s, static_cast<float>(counts[b]) / total);
    }
    e.models_built = true;
}

// ── flushWindow ───────────────────────────────────────────────────────────────

void AionCore::releaseRemovedNodes() {
    // Cells currently occupied by a live DSG node.
    std::unordered_set<size_t> live;
    live.reserve(nodes_cache_.size());
    for (const auto& [id, pos] : nodes_cache_) {
        int64_t ix, iy;
        cellIndices(pos.x(), pos.y(), ix, iy);
        live.insert(gridKey(ix, iy));
    }

    std::vector<size_t> orphaned;
    for (const auto& [k, e] : node_entries_) {
        if (!live.count(k)) orphaned.push_back(k);
    }

    for (size_t k : orphaned) {
        Entry src = std::move(node_entries_[k]);
        node_entries_.erase(k);

        bool fresh = (hash_cells_.find(k) == hash_cells_.end());
        Entry& dst = hash_cells_[k];
        if (fresh) {
            dst.ix  = src.ix;
            dst.iy  = src.iy;
            dst.pos = src.pos;
        }
        mergeInto(dst, src);

        // Drop bindings that pointed at the released node entry.
        for (auto it = hash_bindings_.begin(); it != hash_bindings_.end();) {
            if (it->second == k) it = hash_bindings_.erase(it);
            else ++it;
        }
    }
}

void AionCore::flushWindow(int64_t window_end_ns) {
    uint32_t t_s = static_cast<uint32_t>(window_end_ns / 1'000'000'000LL);

    // 1. Commit each entry's current window into its raw-count history.
    for (auto& [k, e] : node_entries_) commitWindow(e, t_s);
    for (auto& [k, e] : hash_cells_)   commitWindow(e, t_s);

    // 2. Release vanished bound nodes back to hash space (per window).
    releaseRemovedNodes();

    // 3. Bind unbound hash cells to the nearest live DSG node.
    std::vector<size_t> to_erase;
    for (auto& [cell_key, cell] : hash_cells_) {
        if (!cell.hasData()) continue;  // nothing accumulated yet

        auto [cx, cy] = cellCenter(cell.ix, cell.iy);
        auto node_opt = findNearestNode(cx, cy);
        if (!node_opt) continue;  // no DSG node nearby — keep unbound (queryable)

        auto [node_key, node_pos] = *node_opt;

        bool fresh = (node_entries_.find(node_key) == node_entries_.end());
        Entry& ne = node_entries_[node_key];
        if (fresh) {
            int64_t nix, niy;
            cellIndices(node_pos.x(), node_pos.y(), nix, niy);
            ne.ix  = nix;
            ne.iy  = niy;
            ne.pos = node_pos;
        }
        mergeInto(ne, cell);
        hash_bindings_[cell_key] = node_key;
        to_erase.push_back(cell_key);
    }
    for (size_t k : to_erase) hash_cells_.erase(k);
}

// ── logProb ───────────────────────────────────────────────────────────────────

const AionCore::Entry* AionCore::findNearestEntry(double x, double y) const {
    const Entry* best = nullptr;
    double best_d2 = config_.assoc_radius * config_.assoc_radius;

    auto scan = [&](const std::unordered_map<size_t, Entry>& m) {
        for (const auto& [k, e] : m) {
            if (!e.hasData()) continue;
            double dx = e.pos.x() - x, dy = e.pos.y() - y;
            double d2 = dx * dx + dy * dy;
            if (d2 < best_d2) { best_d2 = d2; best = &e; }
        }
    };
    scan(node_entries_);
    scan(hash_cells_);
    return best;
}

double AionCore::logProb(int64_t t_ns, double x, double y, double theta) const {
    const Entry* e = findNearestEntry(x, y);
    if (!e) return std::numeric_limits<double>::quiet_NaN();

    buildModels(*e);

    uint32_t t_s = static_cast<uint32_t>(t_ns / 1'000'000'000LL);

    std::array<double, kNBins> probs;
    double sum = 0.0;
    for (int b = 0; b < kNBins; ++b) {
        probs[b] = std::max(1e-9,
            static_cast<double>(e->models[b].predict(t_s, config_.fremen_order)));
        sum += probs[b];
    }
    return std::log(probs[thetaToBin(theta)] / sum);
}

double AionCore::logProbHeading(int64_t t_ns, double x, double y, double theta) const {
    // p(theta) = P(bin) * kNBins / (2pi): the bin mass spread uniformly across
    // its angular width Δθ = 2pi/kNBins, making it a density over [0, 2pi)
    // a density over [0, 2pi) (the standard p(theta) = P(bin)/Δθ convention).
    const double log_mass = logProb(t_ns, x, y, theta);
    if (std::isnan(log_mass)) return log_mass;
    return log_mass + std::log(static_cast<double>(kNBins) / (2.0 * M_PI));
}

std::vector<double> AionCore::headingDistribution(int64_t t_ns, double x, double y) const {
    // Same lookup, model build, clamp, and normalization as logProb, returning
    // every bin's mass instead of a single bin's log: masses[thetaToBin(theta)]
    // equals exp(logProb(t_ns, x, y, theta)) for any theta.
    const Entry* e = findNearestEntry(x, y);
    if (!e) return {};

    buildModels(*e);

    uint32_t t_s = static_cast<uint32_t>(t_ns / 1'000'000'000LL);

    std::vector<double> probs(kNBins);
    double sum = 0.0;
    for (int b = 0; b < kNBins; ++b) {
        probs[b] = std::max(1e-9,
            static_cast<double>(e->models[b].predict(t_s, config_.fremen_order)));
        sum += probs[b];
    }
    for (int b = 0; b < kNBins; ++b) probs[b] /= sum;
    return probs;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

void AionCore::reset() {
    nodes_cache_.clear();
    hash_cells_.clear();
    node_entries_.clear();
    hash_bindings_.clear();
}

std::size_t AionCore::numTrainedEntries() const {
    std::size_t n = 0;
    for (const auto& [k, e] : node_entries_) if (e.hasData()) ++n;
    for (const auto& [k, e] : hash_cells_)   if (e.hasData()) ++n;
    return n;
}

int AionCore::thetaToBin(double theta) const {
    // Paper convention (03_methodology.tex): bin = floor((theta + pi)/(2*pi/B)) mod B.
    // The +pi zero-reference is a constant relabeling of the slots and does not
    // change results (the same mapping is used to accumulate and to score), but
    // it keeps bin indices identical to the paper's.
    double t = std::fmod(theta + M_PI, 2.0 * M_PI);
    if (t < 0) t += 2.0 * M_PI;
    return static_cast<int>(t / (2.0 * M_PI) * kNBins) % kNBins;
}

// ── Multi-session state I/O ──────────────────────────────────────────────────
//
// Binary layout (version 2):
//   magic[4] = "AION", version: uint32 = 2, schema: string = "aion_state"
//   grid_size: double (must match on load), grid_origin: double[3],
//   world_frame_id: string, fremen_order: int32, n_bins: uint32 (== kNBins)
//   two entry maps (hash cells, then node entries), each:
//     count: uint32
//     for each: key: uint64, ix,iy: int64, pos: double[3],
//               window_counts[kNBins]: int32, window_total: int32,
//               n_history: uint32, for each: (t_s: uint32, counts[kNBins]: int32)

static constexpr char     kAionMagic[4]   = {'A', 'I', 'O', 'N'};
static constexpr uint32_t kAionVersion    = 2;

bool AionCore::saveState(const std::string& path, std::string* error) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        if (error) *error = "cannot open file for writing: " + path;
        return false;
    }
    os.write(kAionMagic, 4);
    writePOD(os, kAionVersion);
    writeString(os, std::string("aion_state"));
    writePOD(os, config_.grid_size);
    double grid_origin[3] = {0.0, 0.0, 0.0};
    os.write(reinterpret_cast<const char*>(grid_origin), sizeof(grid_origin));
    writeString(os, std::string("map"));
    int32_t fremen_order = config_.fremen_order;
    writePOD(os, fremen_order);
    uint32_t nb = static_cast<uint32_t>(kNBins);
    writePOD(os, nb);

    auto write_map = [&](const std::unordered_map<size_t, Entry>& m) {
        uint32_t n = static_cast<uint32_t>(m.size());
        writePOD(os, n);
        for (const auto& [key, e] : m) {
            uint64_t k = static_cast<uint64_t>(key);
            writePOD(os, k);
            writePOD(os, e.ix);
            writePOD(os, e.iy);
            double pos[3] = {e.pos.x(), e.pos.y(), e.pos.z()};
            os.write(reinterpret_cast<const char*>(pos), sizeof(pos));
            for (int b = 0; b < kNBins; ++b) {
                int32_t v = e.window_counts[b];
                writePOD(os, v);
            }
            int32_t total = e.window_total;
            writePOD(os, total);
            uint32_t n_hist = static_cast<uint32_t>(e.history.size());
            writePOD(os, n_hist);
            for (const auto& [t_s, counts] : e.history) {
                writePOD(os, t_s);
                for (int b = 0; b < kNBins; ++b) {
                    int32_t v = counts[b];
                    writePOD(os, v);
                }
            }
        }
    };
    write_map(hash_cells_);
    write_map(node_entries_);

    if (!os.good()) {
        if (error) *error = "write failed: " + path;
        return false;
    }
    return true;
}

bool AionCore::loadState(const std::string& path, std::string* error) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        if (error) *error = "cannot open file for reading: " + path;
        return false;
    }
    char magic[4];
    is.read(magic, 4);
    if (std::memcmp(magic, kAionMagic, 4) != 0) {
        if (error) *error = "magic mismatch (not an AION state file): " + path;
        return false;
    }
    uint32_t version = 0;
    if (!readPOD(is, version)) {
        if (error) *error = "truncated header: " + path;
        return false;
    }
    if (version != kAionVersion) {
        if (error) *error = "version mismatch: file=" + std::to_string(version) +
                            " supported=" + std::to_string(kAionVersion);
        return false;
    }
    std::string schema;
    if (!readString(is, schema) || schema != "aion_state") {
        if (error) *error = "schema mismatch: got '" + schema + "', expected 'aion_state'";
        return false;
    }
    double grid_size = 0.0;
    if (!readPOD(is, grid_size)) {
        if (error) *error = "truncated grid_size";
        return false;
    }
    if (std::abs(grid_size - config_.grid_size) > 1e-9) {
        if (error) *error = "grid_size mismatch: file=" + std::to_string(grid_size) +
                            " current=" + std::to_string(config_.grid_size);
        return false;
    }
    double grid_origin[3];
    is.read(reinterpret_cast<char*>(grid_origin), sizeof(grid_origin));
    std::string frame_id;
    readString(is, frame_id);
    int32_t fremen_order = 0;
    readPOD(is, fremen_order);
    (void)fremen_order;  // informational — models are rebuilt from history on query
    uint32_t nb = 0;
    if (!readPOD(is, nb) || nb != static_cast<uint32_t>(kNBins)) {
        if (error) *error = "n_bins mismatch: file=" + std::to_string(nb) +
                            " kNBins=" + std::to_string(kNBins);
        return false;
    }

    // Wipe current state — fresh DSG run will rebuild bindings naturally.
    reset();

    auto read_map = [&](std::unordered_map<size_t, Entry>& m) -> bool {
        uint32_t n = 0;
        if (!readPOD(is, n)) return false;
        m.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint64_t k = 0;
            if (!readPOD(is, k)) return false;
            Entry e;
            readPOD(is, e.ix);
            readPOD(is, e.iy);
            double pos[3];
            is.read(reinterpret_cast<char*>(pos), sizeof(pos));
            e.pos = Eigen::Vector3d(pos[0], pos[1], pos[2]);
            for (int b = 0; b < kNBins; ++b) {
                int32_t v = 0;
                readPOD(is, v);
                e.window_counts[b] = v;
            }
            int32_t total = 0;
            readPOD(is, total);
            e.window_total = total;
            uint32_t n_hist = 0;
            if (!readPOD(is, n_hist)) return false;
            e.history.reserve(n_hist);
            for (uint32_t j = 0; j < n_hist; ++j) {
                uint32_t t_s = 0;
                readPOD(is, t_s);
                std::array<int, kNBins> counts{};
                for (int b = 0; b < kNBins; ++b) {
                    int32_t v = 0;
                    readPOD(is, v);
                    counts[b] = v;
                }
                e.history.emplace_back(t_s, counts);
            }
            if (!is.good()) return false;
            m.emplace(static_cast<size_t>(k), std::move(e));
        }
        return true;
    };

    if (!read_map(hash_cells_) || !read_map(node_entries_)) {
        if (error) *error = "truncated entry payload: " + path;
        return false;
    }

    // hash_bindings_ intentionally not restored: the previous DSG's NodeIds are
    // stale. Next flushWindow() re-binds via findNearestNode().
    return true;
}

}  // namespace aion
