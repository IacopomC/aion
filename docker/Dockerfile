# Ready-to-run image for the AION baseline (Ubuntu 20.04 / ROS Noetic).
#
# Build from the root of this repository (context must be the repo root):
#   docker build -f docker/Dockerfile -t aion:noetic .
# Run:
#   docker run --rm -it aion:noetic
#   (inside: ROS + the catkin workspace are already sourced)
FROM ros:noetic-perception-focal

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# --- toolchain: newer CMake (Kitware), GTSAM (borglab PPA), nlohmann_json,
#     modern pybind11, and catkin build tools ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates wget gnupg \
    && wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor - > /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
        > /etc/apt/sources.list.d/kitware.list \
    && add-apt-repository -y ppa:borglab/gtsam-release-4.1 \
    && apt-get update && apt-get install -y --no-install-recommends \
        cmake build-essential git \
        python3-pip python3-catkin-tools python3-rosdep python3-dev \
        libeigen3-dev libgoogle-glog-dev libgflags-dev libboost-all-dev \
        nlohmann-json3-dev libgtsam-dev libgtsam-unstable-dev \
        libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "pybind11>=2.11"

# --- workspace dependencies: public repos pinned to verified versions ---
WORKDIR /ws/src
RUN set -e; \
    git clone https://github.com/IacopomC/hydra-aion.git hydra; \
    git clone https://github.com/MIT-SPARK/Hydra-ROS.git hydra_ros    && git -C hydra_ros        checkout d876720;  \
    git clone https://github.com/IacopomC/spark-dsg-aion.git spark_dsg && git -C spark_dsg checkout 28fc8f9 && git -C spark_dsg submodule update --init --recursive; \
    git clone https://github.com/MIT-SPARK/config_utilities.git       && git -C config_utilities checkout archive/ros_noetic; \
    git clone https://github.com/MIT-SPARK/Kimera-PGMO.git kimera_pgmo && git -C kimera_pgmo      checkout archive/ros_noetic; \
    git clone https://github.com/MIT-SPARK/Kimera-RPGO.git kimera_rpgo && git -C kimera_rpgo      checkout d28b4df0570d642a2bb00e511344ce1110f87519; \
    git clone https://github.com/MIT-SPARK/pose_graph_tools.git        && git -C pose_graph_tools checkout archive/ros_noetic; \
    git clone https://github.com/MIT-SPARK/Spatial-Hash.git spatial_hash && git -C spatial_hash   checkout 88b235966c998ec97cd910dece1c70139235fec1; \
    git clone https://github.com/MIT-SPARK/TEASER-plusplus.git teaser_plusplus && git -C teaser_plusplus checkout f91cfdb7baed951a3607257bd31f3f6694773497; \
    git clone https://github.com/IacopomC/libfremen.git libfremen

# --- Serialize GTSAM's TBB multifrontal elimination. Its parallel back-
#     substitution (OptimizeClique) races on the dense deformation-graph
#     structure built up under many loop closures, throwing std::out_of_range
#     ("map::at") or segfaulting nondeterministically. The solve is tiny
#     (<1k variables, a few ms) so serial elimination costs nothing measurable.
#     Kimera-RPGO is upstream MIT-SPARK (cannot be patched via commit), so the
#     guard is injected into the pinned clone here. See HANDOFF_gtsam_tbb_crash.md.
RUN set -e; cd /ws/src/kimera_rpgo/src; \
    sed -i 's@#include <gtsam/slam/dataset.h>@&\n#include <tbb/global_control.h>@' RobustSolver.cpp; \
    sed -i 's@^void RobustSolver::optimize() {@&\n  tbb::global_control gtsam_serial(tbb::global_control::max_allowed_parallelism, 1);@' RobustSolver.cpp; \
    grep -q 'tbb::global_control gtsam_serial' RobustSolver.cpp \
      || { echo "ERROR: Kimera-RPGO TBB-serial patch failed to apply (upstream anchor changed?)"; exit 1; }

# --- the contribution (this repository; includes the fremenarray package) ---
COPY . /ws/src/aion

# --- build (Release). `aion` declares `hydra` as a dependency, so catkin
#     builds hydra, fremenarray and the rest in the right order. ---
WORKDIR /ws
RUN source /opt/ros/noetic/setup.bash \
    && catkin config --extend /opt/ros/noetic \
         -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
         -DCONFIG_UTILS_BUILD_DEMOS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && catkin build aion

# --- python: install the spark-dsg-aion fork's package (it carries the
#     save-DSG binding the aion pipeline relies on), then the aion bindings.
#     The aion bindings are editable: their CMake pulls pybind11 from
#     ../../../spark_dsg and links fremen_core from ../../../libfremen via
#     relative paths, which only resolve for an in-place editable build. ---
RUN source /opt/ros/noetic/setup.bash && source /ws/devel/setup.bash \
    && pip3 install /ws/src/spark_dsg \
    && pip3 install -e /ws/src/aion/aion/python

# --- hydra_python: the offline runner drives the Hydra pipeline
#     (FileDataLoader + load_pipeline + pipeline.step). Build the bindings
#     against the catkin-built libs. --no-deps is REQUIRED so pip does not pull
#     stock spark_dsg from GitHub and clobber the spark-dsg-aion fork installed
#     above; the light runtime deps are added explicitly (spark_dsg excluded). ---
RUN source /opt/ros/noetic/setup.bash && source /ws/devel/setup.bash \
    && cp /ws/src/aion/aion/config/hydra_tbd.yaml /ws/src/hydra/config/datasets/tbd.yaml \
    && sed -i 's|GIT_TAG master|GIT_TAG v2.13.6|' /ws/src/hydra/python/CMakeLists.txt \
    && pip3 install --no-deps /ws/src/hydra \
    && pip3 install click distinctipy imageio networkx pyqtgraph scipy tqdm \
    && pip3 install -U "numpy==1.24.4"
# tbd.yaml (the TBD Hydra dataset config) is upstream-absent; copy our
# version-controlled copy into hydra/config/datasets BEFORE the pip install so
# it is bundled into the installed hydra_python config (load_configs finds it,
# no runtime mount). Must precede the hydra install.
# The sed pins Hydra's pybind11 (default GIT_TAG master) to v2.13.6 to MATCH the
# version spark_dsg vendors. Mismatched pybind11 → incompatible cross-module
# `internals` → pipeline.graph returns an "Unregistered type spark_dsg::
# DynamicSceneGraph". Same version → shared registry → the runner reads layer-20.
# numpy bumped last: hydra_python eagerly imports its simulators -> spark_dsg.mp3d
# -> seaborn/matplotlib, which need numpy>=1.20 (the base image ships 1.17.4).
# 1.24.4 is the last py3.8 release and satisfies matplotlib/scipy/pandas/pyqtgraph;
# the pre-built C++ bindings keep working (numpy C-API is backward compatible).

# Entrypoint: source ROS + the catkin workspace (so libspark_dsg.so etc. are on
# the loader path) and expose the aion_python bindings on PYTHONPATH, for every
# command (interactive or not). Makes `docker run aion:noetic python3 ...` work.
RUN printf '%s\n' '#!/bin/bash' \
      'source /opt/ros/noetic/setup.bash' \
      'source /ws/devel/setup.bash' \
      'export PYTHONPATH=/ws/src/aion/aion/python:${PYTHONPATH:-}' \
      'exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh
WORKDIR /ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
