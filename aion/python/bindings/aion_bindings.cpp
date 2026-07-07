#include "aion_core.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace aion;

PYBIND11_MODULE(_aion_standalone, m) {
    m.doc() = "AION standalone pybind11 module — no ROS";

    py::class_<AionConfig>(m, "AionConfig")
        .def(py::init<>())
        .def_readwrite("assoc_radius",     &AionConfig::assoc_radius)
        .def_readwrite("grid_size",        &AionConfig::grid_size)
        .def_readwrite("fremen_order",     &AionConfig::fremen_order)
        .def_readwrite("candidate_periods",&AionConfig::candidate_periods)
        .def_readwrite("max_detection_range_m", &AionConfig::max_detection_range_m,
                       "Drop detections farther than this (m) from the robot pose. "
                       "inf = disabled.");

    py::class_<AionCore>(m, "AionCore")
        .def(py::init<AionConfig>(), py::arg("config") = AionConfig{})
        .def("update_node_positions", &AionCore::updateNodePositions,
             py::arg("nodes"),
             "Replace DSG node cache with current layer-20 snapshot. "
             "nodes: list of (node_id: int, x, y, z). "
             "Call after every pipeline.step() during training.")
        .def("add_detection", &AionCore::addDetection,
             py::arg("t_ns"), py::arg("x"), py::arg("y"), py::arg("theta"),
             py::arg("robot_x") = 0.0, py::arg("robot_y") = 0.0,
             "Accumulate one detection into the hash cell (unbound) or "
             "node entry (already bound) for this position.\n"
             "robot_x/robot_y gate detections by max_detection_range_m.")
        .def("flush_window", &AionCore::flushWindow,
             py::arg("window_end_ns"),
             "End-of-window step: flush trained node entries to FreMEn, "
             "then try to bind unbound hash cells to nearby DSG nodes.")
        .def("log_prob", &AionCore::logProb,
             py::arg("t_ns"), py::arg("x"), py::arg("y"), py::arg("theta"),
             "log P(heading | nearest trained node entry, t), or NaN.")
        .def("log_prob_heading", &AionCore::logProbHeading,
             py::arg("t_ns"), py::arg("x"), py::arg("y"), py::arg("theta"),
             "Heading log-density log p(theta) = log P(bin) + log(kNBins/2pi),\n"
             "the bin mass as a density over [0, 2pi). NaN if no trained cell.")
        .def("heading_distribution", &AionCore::headingDistribution,
             py::arg("t_ns"), py::arg("x"), py::arg("y"),
             "Normalized per-bin heading masses at the query point and time,\n"
             "in bin index order; empty list if no trained cell within\n"
             "assoc_radius. masses[theta_to_bin(theta)] == exp(log_prob(...)),\n"
             "so one call serves every bin.")
        .def("theta_to_bin", &AionCore::thetaToBin,
             py::arg("theta"),
             "Bin index for a heading (the same mapping used to accumulate "
             "and score).")
        .def("reset", &AionCore::reset,
             "Clear all state including bindings (use between full experiments, "
             "not between sessions — bindings persist across sessions by design).")
        .def("save_state",
             [](const AionCore& c, const std::string& p) {
                 std::string err;
                 bool ok = c.saveState(p, &err);
                 return py::make_tuple(ok, err);
             },
             py::arg("path"),
             "Persist hash cells + node entries (with FreMEn observation history) "
             "to disk. Returns (ok: bool, error: str).")
        .def("load_state",
             [](AionCore& c, const std::string& p) {
                 std::string err;
                 bool ok = c.loadState(p, &err);
                 return py::make_tuple(ok, err);
             },
             py::arg("path"),
             "Restore hash cells + node entries from a prior session. "
             "Aborts on grid_size/n_bins mismatch. Returns (ok, error). "
             "Call BEFORE driving the new session's pipeline; previous bindings "
             "are discarded so the new DSG run can re-bind naturally.")
        .def_property_readonly("num_hash_cells",    &AionCore::numHashCells,
             "Unbound hash cells still waiting for a nearby DSG node.")
        .def_property_readonly("num_node_entries",  &AionCore::numNodeEntries,
             "Bound node entries (have at least one FreMEn observation).")
        .def_property_readonly("num_trained_entries",&AionCore::numTrainedEntries);
}
