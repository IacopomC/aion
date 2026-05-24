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
        .def_readwrite("candidate_periods",&AionConfig::candidate_periods);

    py::class_<AionCore>(m, "AionCore")
        .def(py::init<AionConfig>(), py::arg("config") = AionConfig{})
        .def("update_node_positions", &AionCore::updateNodePositions,
             py::arg("nodes"),
             "Replace DSG node cache with current layer-20 snapshot. "
             "nodes: list of (node_id: int, x, y, z). "
             "Call after every pipeline.step() during training.")
        .def("add_detection", &AionCore::addDetection,
             py::arg("t_ns"), py::arg("x"), py::arg("y"), py::arg("theta"),
             "Accumulate one detection into the hash cell (unbound) or "
             "node entry (already bound) for this position.")
        .def("flush_window", &AionCore::flushWindow,
             py::arg("window_end_ns"),
             "End-of-window step: flush trained node entries to FreMEn, "
             "then try to bind unbound hash cells to nearby DSG nodes.")
        .def("log_prob", &AionCore::logProb,
             py::arg("t_ns"), py::arg("x"), py::arg("y"), py::arg("theta"),
             "log P(heading | nearest trained node entry, t), or NaN.")
        .def("reset", &AionCore::reset,
             "Clear all state including bindings (use between full experiments, "
             "not between sessions — bindings persist across sessions by design).")
        .def_property_readonly("num_hash_cells",    &AionCore::numHashCells,
             "Unbound hash cells still waiting for a nearby DSG node.")
        .def_property_readonly("num_node_entries",  &AionCore::numNodeEntries,
             "Bound node entries (have at least one FreMEn observation).")
        .def_property_readonly("num_trained_entries",&AionCore::numTrainedEntries);
}
